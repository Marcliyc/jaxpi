from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

from flax.training import train_state
from flax import jax_utils

import jax.numpy as jnp
from jax import lax, jit, grad, pmap, random, jacfwd, jacrev
from jax.tree_util import tree_map, tree_reduce, tree_leaves
from flax.traverse_util import flatten_dict, unflatten_dict


import optax

from soap_jax import soap  # Install from https://github.com/haydn-jones/SOAP_JAX
from jaxpi import archs
from jaxpi.utils import flatten_pytree


class TrainState(train_state.TrainState):
    weights: Dict
    momentum: float

    def apply_weights(self, weights, **kwargs):
        """Updates `weights` using running average  in return value.

        Returns:
          An updated instance of `self` with new weights updated by applying `running_average`,
          and additional attributes replaced as specified by `kwargs`.
        """

        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree_map(running_average, self.weights, weights)
        weights = lax.stop_gradient(weights)

        return self.replace(
            step=self.step,
            params=self.params,
            opt_state=self.opt_state,
            weights=weights,
            **kwargs,
        )


def _create_arch(config):
    if config.arch_name == "Mlp":
        arch = archs.Mlp(**config)

    elif config.arch_name == "ModifiedMlp":
        arch = archs.ModifiedMlp(**config)

    elif config.arch_name == "DeepONet":
        arch = archs.DeepONet(**config)

    elif config.arch_name == "TimeDependentPINN":
        arch = archs.TimeDependentPINN(**config)
    
    elif config.arch_name == "PINN_Gaussian":
        arch = archs.PINN_Gaussian(**config)

    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def _create_optimizer(config):

    staircase = config.get("staircase", False)
    lr = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.decay_steps,
            decay_rate=config.decay_rate,
            staircase=staircase
        )
    
    if config.warmup_steps > 0:
        warmup = optax.linear_schedule(init_value=0.0, end_value=config.learning_rate,
                                        transition_steps=config.warmup_steps)

        lr = optax.join_schedules([warmup, lr], [config.warmup_steps])

    if config.optimizer == "Adam":
        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )
    elif config.optimizer == "Soap":
        tx = soap(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, weight_decay=0.0, precondition_frequency=2
            )
        
    elif config.optimizer == "Muon":
        tx = optax.contrib.muon(
            learning_rate=lr,
            ns_coeffs=(2, -1.5, 0.5),
            ns_steps=10,
            beta=0.99,
            adam_b1=0.99
        )
    
    elif config.optimizer == 'ClipAdamW':
        lr = optax.warmup_cosine_decay_schedule(
            init_value=config.learning_rate * 0.01,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
            end_value=config.learning_rate * 0.01,
        )
        # lr = optax.exponential_decay(
        #     init_value=config.learning_rate,
        #     transition_steps=config.decay_steps,
        #     decay_rate=config.decay_rate,
        # )
        def make_mask_gaussian(params):
            flat = flatten_dict(params, sep="/")
            mask_flat = {}
            for k in flat.keys():
                name = k
                # Example: don't decay Gaussian centers/sigmas
                if name.endswith("/mu") or name.endswith("/log_sigma") or name.endswith("/sigmas"):
                    mask_flat[k] = False
                # still skip biases/scales
                elif name.endswith("/bias") or name.endswith("/scale"):
                    mask_flat[k] = False
                else:
                    mask_flat[k] = True
            return unflatten_dict(mask_flat, sep="/")

        # Optimizer with gradient clipping
        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adamw(lr, b1=config.beta1,b2=config.beta2,eps=config.eps,weight_decay=config.weight_decay, mask=make_mask_gaussian),
        )
    
    elif config.optimizer == 'ClipAdam':
        lr = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.decay_steps,
            decay_rate=config.decay_rate,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adam(learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps),
        )

    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")

    # Gradient accumulation
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)

    return tx


def _create_train_state(config):
    # Initialize network
    arch = _create_arch(config.arch)
    x = jnp.ones(config.input_dim)
    if 'Time' not in arch.arch_name:
        params = arch.init(random.PRNGKey(config.seed), x)
    else:
        # For TimeDependentPINN which requires two inputs
        t = 1.0
        params = arch.init(random.PRNGKey(config.seed), x, t)

    # Initialize optax optimizer
    tx = _create_optimizer(config.optim)

    # Convert config dict to dict
    init_weights = dict(config.weighting.init_weights)

    state = TrainState.create(
        apply_fn=arch.apply,
        params=params,
        tx=tx,
        weights=init_weights,
        momentum=config.weighting.momentum,
    )

    return jax_utils.replicate(state)


class PINN:
    def __init__(self, config):
        self.config = config
        self.state = _create_train_state(config)

    def u_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def losses(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def compute_diag_ntk(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, *args):
        # Compute losses
        losses = self.losses(params, batch, *args)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)

        loss= jnp.where(jnp.isnan(loss), 1e6, loss)
        return loss

    @partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch, *args):
        if self.config.weighting.scheme == "grad_norm":
            # Compute the gradient of each loss w.r.t. the parameters
            grads = jacrev(self.losses)(params, batch, *args)

            # Compute the grad norm of each loss
            grad_norm_dict = {}
            for key, value in grads.items():
                flattened_grad = flatten_pytree(value)
                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

            # Compute the mean of grad norms over all losses
            mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
            # Grad Norm Weighting
            w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)

        elif self.config.weighting.scheme == "ntk":
            # Compute the diagonal of the NTK of each loss
            ntk = self.compute_diag_ntk(params, batch, *args)

            # Compute the mean of the diagonal NTK corresponding to each loss
            mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)

            # Compute the average over all ntk means
            mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_dict)))
            # NTK Weighting
            w = tree_map(lambda x: (mean_ntk / x), mean_ntk_dict)

        return w

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch, *args):
        weights = self.compute_weights(state.params, batch, *args)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step(self, state, batch, *args):
        grads = grad(self.loss)(state.params, state.weights, batch, *args)
        grads = tree_map(
            lambda g: jnp.where(jnp.isnan(g), 0.0, g), 
            grads
        )
        grads = lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step_with_grad_stats(self, state, batch, *args):
        grads = grad(self.loss)(state.params, state.weights, batch, *args)
        grads = tree_map(
            lambda g: jnp.where(jnp.isnan(g), 0.0, g),
            grads,
        )
        grads = lax.pmean(grads, "batch")

        flat_grads = flatten_dict(grads, sep="/")
        grad_norms = [jnp.linalg.norm(flatten_pytree(g)) for g in flat_grads.values()]
        grad_norms = jnp.stack(grad_norms)
        # max_grad_idx = jnp.argmax(grad_norms)
        # max_grad_norm = grad_norms[max_grad_idx]

        state = state.apply_gradients(grads=grads)
        return state, grad_norms #max_grad_norm, max_grad_idx

    def get_grad_layer_names(self):
        params = tree_map(lambda x: x[0], self.state.params)
        flat_params = flatten_dict(params, sep="/")
        return list(flat_params.keys())


class ForwardIVP(PINN):
    def __init__(self, config):
        super().__init__(config)

        if config.weighting.use_causal:
            self.tol = config.weighting.causal_tol
            self.num_chunks = config.weighting.num_chunks
            self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T


class ForwardBVP(PINN):
    def __init__(self, config):
        super().__init__(config)
