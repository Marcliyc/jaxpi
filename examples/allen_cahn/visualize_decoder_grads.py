#!/usr/bin/env python3
"""Visualize gradient norms w.r.t. TimeConditionedDecoder hidden states."""
import argparse
import importlib.util
import os

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jaxpi import archs
from jaxpi import models as core_models
from jaxpi.utils import restore_checkpoint


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"Config file {config_path} must define get_config().")
    return module.get_config()


def unreplicate_state(state):
    leaf = jax.tree.leaves(state.params)[0]
    if getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] in (1, jax.local_device_count()):
        state = state.replace(params=tree_map(lambda x: x[0], state.params))
    return state


def get_x_range(config):
    x_min = None
    x_max = None
    if hasattr(config, "arch") and hasattr(config.arch, "pyramid"):
        x_min = config.arch.pyramid.get("x_min", None)
        x_max = config.arch.pyramid.get("x_max", None)
    if x_min is None or x_max is None:
        x_min, x_max = -1.0, 1.0
    return float(x_min), float(x_max)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize gradient norms w.r.t. hidden states in TimeConditionedDecoder."
    )
    parser.add_argument(
        "--config",
        default="jaxpi/examples/allen_cahn/configs/grid11d_tanh_reparam0.py",
        help="Path to config .py file.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to checkpoint directory. If omitted, uses ./<wandb.name>/ckpt when available.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest).",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.0,
        help="Time value for evaluation.",
    )
    parser.add_argument(
        "--num_x",
        type=int,
        default=64,
        help="Number of x points to sample.",
    )
    parser.add_argument(
        "--plot",
        choices=["line", "heatmap", "both"],
        default="both",
        help="Plot type to generate.",
    )
    parser.add_argument(
        "--out",
        default="decoder_grad_norms.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Initialize model state (shape only); no dataset needed for gradients.
    state = core_models._create_train_state(config)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        default_ckpt = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
        if os.path.isdir(default_ckpt):
            ckpt_path = default_ckpt

    if ckpt_path and os.path.isdir(ckpt_path):
        state = restore_checkpoint(state, ckpt_path, step=args.step)
        state = unreplicate_state(state)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        state = unreplicate_state(state)
        if ckpt_path:
            print(f"Checkpoint not found at {ckpt_path}. Using initialized weights.")
        else:
            print("No checkpoint provided/found. Using initialized weights.")

    params = state.params

    x_min, x_max = get_x_range(config)
    xs = np.linspace(x_min, x_max, args.num_x)
    t = float(args.t)

    # Get number of levels from a single forward pass
    x0 = jnp.array([xs[0]])
    _, states0 = state.apply_fn(
        params,
        x0,
        t,
        method=archs.TimeDependentPINN.forward_with_states,
    )
    n_levels = len(states0)

    grad_norms = np.zeros((n_levels, args.num_x), dtype=np.float32)

    for i, x in enumerate(xs):
        x_arr = jnp.array([x])
        _, states = state.apply_fn(
            params,
            x_arr,
            t,
            method=archs.TimeDependentPINN.forward_with_states,
        )

        for k, h in enumerate(states):
            def out_from_h(h_in):
                out = state.apply_fn(
                    params,
                    x_arr,
                    t,
                    h_in,
                    k,
                    method=archs.TimeDependentPINN.decode_from,
                )
                return jnp.sum(out)

            grad_h = jax.grad(out_from_h)(h)
            grad_norms[k, i] = float(jnp.linalg.norm(grad_h))

    ncols = 2 if args.plot == "both" else 1
    fig, axes = plt.subplots(1, ncols, figsize=(12 if ncols == 2 else 8, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    idx = 0
    if args.plot in ("line", "both"):
        mean_norm = grad_norms.mean(axis=1)
        axes[idx].plot(np.arange(n_levels), mean_norm, marker="o")
        axes[idx].set_xlabel("Decoder level")
        axes[idx].set_ylabel("Mean grad norm")
        axes[idx].set_title("Gradient norm by level")
        idx += 1

    if args.plot in ("heatmap", "both"):
        im = axes[idx].imshow(
            grad_norms,
            aspect="auto",
            origin="lower",
            extent=[x_min, x_max, 0, n_levels - 1],
            cmap="viridis",
        )
        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("Decoder level")
        axes[idx].set_title("Grad norm heatmap")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.02)

    fig.suptitle("Gradient Norms w.r.t. Decoder Hidden States", y=1.02)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
