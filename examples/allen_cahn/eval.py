import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

import jax
from jax.tree_util import tree_map

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0, :]

    if config.get("use_pi_init", False):
        if config.arch.arch_name == "TimeDependentPINN":
            config.arch.pi_init = jnp.zeros((64, config.arch.out_dim))
        elif config.arch.arch_name == "PINN_Gaussian":
            config.arch.pi_init = jnp.zeros((config.arch.features[-1], config.arch.out_dim))
        elif hasattr(config.arch, "hidden_dim"):
            config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    # Restore model
    if config.arch.arch_name == "TimeDependentPINN":
        model = models.AllenCahnTime(config, u0, t_star, x_star)
    else:
        model = models.AllenCahn(config, u0, t_star, x_star)
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    #ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    state = restore_checkpoint(model.state, ckpt_path)
   
    leaf = jax.tree.leaves(state.params)[0]
    if leaf.ndim > 0 and leaf.shape[0] in (1, jax.local_device_count()):
            state = state.replace(params=tree_map(lambda x: x[0], state.params))
            # or: state = state.replace(params=unreplicate(state.params))  # if it’s standard replicated
    model.state = state
    params = model.state.params
    l2_error = model.compute_l2_error(params, u_ref)

    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")

    # plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Reference")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "ac.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
