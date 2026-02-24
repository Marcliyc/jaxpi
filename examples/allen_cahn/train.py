import os
import time

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map

import scipy.io

import ml_collections
import wandb


from jaxpi.archs import PeriodEmbs, Embedding
from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Get dataset
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0, :]

    t0 = t_star[0]
    t1 = t_star[-1]

    x0 = x_star[0]
    x1 = x_star[-1]

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Define residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    if config.get("use_pi_init", False):
        logger.info("Use physics-informed initialization...")

        model = models.AllenCahn(config, u0, t_star, x_star)
        state = jax.device_get(tree_map(lambda x: x[0], model.state))
        params = state.params

        # Initialization data source
        if config.pi_init_type == "linear_pde":
            # load data
            data = scipy.io.loadmat("data/allen_cahn_linear.mat")
            # downsample the grid and data
            u = data["usol"][::10]
            t = data["t"].flatten()[::10]
            x = data["x"].flatten()

            tt, xx = jnp.meshgrid(t, x, indexing="ij")
            inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])

        elif config.pi_init_type == "initial_condition":
            t = t_star[::10]
            x = x_star
            u = u0

            tt, xx = jnp.meshgrid(t, x, indexing="ij")
            inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])
            u = jnp.tile(u.flatten(), (t.shape[0], 1))

        if config.arch.arch_name == "TimeDependentPINN":
            feat_matrix, _ = vmap(
                lambda z: state.apply_fn(params, z[1:], z[0]),
                (0,),
            )(inputs)
        else:
            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

        coeffs, residuals, rank, s = jnp.linalg.lstsq(
            feat_matrix, u.flatten(), rcond=None
        )
        print("least square residuals: ", residuals)

        config.arch.pi_init = coeffs.reshape(
            -1, 1
        )  # Be careful, this overwrites the config file!

        del model, state, params

    # Initialize model
    if 'Time' in config.arch.arch_name:
        model = models.AllenCahnTime(config, u0, t_star, x_star)
    else:
        model = models.AllenCahn(config, u0, t_star, x_star)

    # Initialize evaluator
    evaluator = models.AllenCanhEvaluator(config, model)

    grad_layer_names = model.get_grad_layer_names()

    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)

        if config.logging.log_grads and step % config.logging.log_every_steps == 0:
            #model.state, max_grad_norm, max_grad_idx = model.step_with_grad_stats(model.state, batch)
            model.state, grad_norms = model.step_with_grad_stats(model.state, batch)
        else:
            model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)

                if config.logging.log_grads:
                    # max_grad_norm_host = float(jax.device_get(max_grad_norm)[0])
                    # max_grad_idx_host = int(jax.device_get(max_grad_idx)[0])
                    # log_dict["max_layer_grad_norm"] = max_grad_norm_host
                    # log_dict["max_layer_grad_idx"] = max_grad_idx_host
                    # log_dict["max_layer_grad_name"] = grad_layer_names[max_grad_idx_host]
                    grad_norms_host = jax.device_get(grad_norms)[0]
                    for name, norm in zip(grad_layer_names, grad_norms_host):
                        log_dict[f"{name}_grad_norm"] = float(norm)
                    log_dict['max_grad_layer'] = grad_layer_names[jnp.argmax(grad_norms_host)]
                    log_dict['max_grad_norm'] = float(jnp.max(grad_norms_host))

                wandb.log(log_dict, step)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model
