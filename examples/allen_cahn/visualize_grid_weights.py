#!/usr/bin/env python3
"""Visualize 1-D grid weights for TimeDependentPINN (multi-res pyramid)."""
import argparse
import os
import importlib.util

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map

from flax.traverse_util import flatten_dict

from jaxpi.archs import interpolate_grid_nd
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


def find_grid_params(params):
    flat = flatten_dict(params, sep="/")
    grids = {}
    for key, value in flat.items():
        name = key.split("/")[-1]
        if name.startswith("grid_"):
            grids[key] = value
    return grids


def level_from_key(key: str) -> int:
    name = key.split("/")[-1]
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def sort_grid_items(grid_items):
    return sorted(grid_items.items(), key=lambda kv: level_from_key(kv[0]))


def compute_x_axis(config, num_x: int):
    pyramid = config.arch.pyramid
    x_min = pyramid.get("x_min", None)
    x_max = pyramid.get("x_max", None)
    if x_min is None or x_max is None:
        return np.linspace(0.0, 1.0, num_x), None, None

    attn_mode = pyramid.get("attn_mode", (0,))
    periodic = all(m == 1 for m in attn_mode)
    # For periodic grids, avoid duplicating the endpoint
    x = np.linspace(x_min, x_max, num_x, endpoint=not periodic)
    return x, float(x_min), float(x_max)


def interpolate_features_1d(grid, x_points, res, attn_mode, x_min, x_max):
    # Evaluate feature vectors using the same cubic B-spline interpolation as the model.
    x_jax = jnp.asarray(x_points)

    def eval_at_x(xi):
        return interpolate_grid_nd(
            grid=grid,
            x=jnp.asarray([xi]),
            res=res,
            attn_mode=attn_mode,
            x_min=x_min,
            x_max=x_max,
        )

    return np.array(vmap(eval_at_x)(x_jax))


def main():
    parser = argparse.ArgumentParser(description="Visualize 1-D grid weights for TimeDependentPINN.")
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
        "--num_x",
        type=int,
        default=512,
        help="Number of linearly spaced x points used for interpolation.",
    )
    parser.add_argument(
        "--plot",
        choices=["stat", "feature_lines", "both", "heatmap"],
        default="both",
        help="`stat`: summary line, `feature_lines`: one line per feature dim. "
             "`heatmap` is kept as an alias for `feature_lines`.",
    )
    parser.add_argument(
        "--stat",
        choices=["l2", "mean", "max"],
        default="l2",
        help="Statistic for line plots across feature dim.",
    )
    parser.add_argument(
        "--out",
        default="grid_weights.png",
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

    # Initialize model state (shape only); no dataset needed for weights.
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

    grid_params = find_grid_params(state.params)
    if not grid_params:
        raise RuntimeError("No grid_* parameters found. Is the config using a pyramid? ")

    grid_items = sort_grid_items(grid_params)
    pyramid = config.arch.pyramid
    base_res = int(pyramid.base_resolution)
    attn_mode = tuple(int(m) for m in pyramid.attn_mode)
    if len(attn_mode) != 1:
        raise ValueError(
            f"This script currently supports 1-D only. Got attn_mode={attn_mode}."
        )

    show_stat = args.plot in ("stat", "both")
    show_feature_lines = args.plot in ("feature_lines", "both", "heatmap")

    n_levels = len(grid_items)
    ncols = int(show_stat) + int(show_feature_lines)
    if ncols == 0:
        raise RuntimeError("No plot selected.")
    figsize = (12, max(2.2 * n_levels, 3.0)) if ncols == 2 else (10, max(2.2 * n_levels, 3.0))
    fig, axes = plt.subplots(nrows=n_levels, ncols=ncols, figsize=figsize, constrained_layout=True)
    if n_levels == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(n_levels, 1)

    for row, (key, grid) in enumerate(grid_items):
        level = level_from_key(key)
        res = base_res * (2 ** level)
        x, x_min, x_max = compute_x_axis(config, args.num_x)
        feature_values = interpolate_features_1d(
            grid=grid,
            x_points=x,
            res=res,
            attn_mode=attn_mode,
            x_min=x_min,
            x_max=x_max,
        )
        feat_dim = feature_values.shape[-1]

        title = f"{key} | res={res} | feat={feat_dim}"

        col = 0
        if show_stat:
            if args.stat == "l2":
                stat = np.linalg.norm(feature_values, axis=-1)
                stat_name = "L2"
            elif args.stat == "mean":
                stat = np.mean(feature_values, axis=-1)
                stat_name = "Mean"
            else:
                stat = np.max(feature_values, axis=-1)
                stat_name = "Max"
            ax = axes[row, col]
            ax.plot(x, stat, linewidth=1.6)
            ax.set_title(f"{title} | {stat_name}")
            ax.set_xlabel("x" if x_min is not None else "grid index")
            ax.set_ylabel("weight")
            col += 1

        if show_feature_lines:
            ax = axes[row, col]
            for feat_idx in range(feat_dim):
                ax.plot(x, feature_values[:, feat_idx], linewidth=0.8, alpha=0.7)
            ax.set_title(f"{title} | interpolated per-feature lines")
            ax.set_xlabel("x" if x_min is not None else "grid index")
            ax.set_ylabel("weight value")

    fig.suptitle("1-D Grid Weights (TimeDependentPINN Pyramid)", y=1.02)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
