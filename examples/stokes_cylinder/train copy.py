import time
import os

from absl import logging

import jax
import jax.numpy as jnp
from jax import vmap, jacrev
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import ml_collections

import wandb

import matplotlib.pyplot as plt

from jaxpi.samplers import SpaceSampler, SpaceSamplerCheat
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset, parabolic_inflow


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    logger = Logger()

    # Get dataset
    (
        u_ref,  
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,  
        cylinder_coords,
        nu,
    ) = get_dataset(cheat=config.cheat)

    # Inflow boundary conditions
    U_max = 0.3  # maximum velocity
    u_inflow, _ = parabolic_inflow(inflow_coords[:, 1], U_max)

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = 0.2  # characteristic velocity
        L_star = 0.1  # characteristic length
        Re = U_star * L_star / nu

        # Nondimensionalize coordinates and inflow velocity
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        wall_coords = wall_coords / L_star
        cylinder_coords = cylinder_coords / L_star
        coords = coords / L_star

        # Nondimensionalize flow field
        u_inflow = u_inflow / U_star
        u_ref = u_ref / U_star
        v_ref = v_ref / U_star
        if config.cheat:
            p_ref, p_x_ref, p_y_ref, p_out,p_x_out,p_y_out = p_ref
            p_ref = p_ref / U_star**2
            p_out   = p_out  / U_star**2
            dp_scale = L_star / U_star**2
            p_x_ref   = p_x_ref * dp_scale
            p_y_ref   = p_y_ref * dp_scale
            p_x_out   = p_x_out * dp_scale
            p_y_out   = p_y_out * dp_scale


    else:
        U_star = 1.0
        L_star = 1.0
        Re = 1 / nu

    # Initialize model
    if not config.cheat:
        if config.bc_constraints == "hard":
            model = models.Stokes2DHardAll(
                config,
                u_inflow,
                inflow_coords,
                outflow_coords,
                wall_coords,
                cylinder_coords,
                Re,
                hard_outflow=True,
            )
        elif config.bc_constraints == "hybrid":
            model = models.Stokes2DHardBC(
                config,
                u_inflow,
                inflow_coords,
                outflow_coords,
                wall_coords,
                cylinder_coords,
                Re,
                cylinder_dist = config.cylinder_dist,
                wall_dist= config.wall_dist,
            )
        else:
            model = models.Stokes2D(
                config,
                u_inflow,
                inflow_coords,
                outflow_coords,
                wall_coords,
                cylinder_coords,
                Re,
            )
    else:
        if config.bc_constraints == "hard":
            model = models.Stokes2DHardAllCheat(
                config,
                u_inflow,
                inflow_coords,
                outflow_coords,
                wall_coords,
                cylinder_coords,
                Re,
                p_out,
                p_x_out,
                p_y_out,
                hard_outflow=True,
            )
        elif config.bc_constraints == "hybrid":
            model = models.Stokes2DHardBCCheat(
                config,
                u_inflow,
                inflow_coords,
                outflow_coords,
                wall_coords,
                cylinder_coords,
                Re,
                p_out,
                p_x_out,
                p_y_out,
                cylinder_dist = config.cylinder_dist,
                wall_dist= config.wall_dist,
            )
        else:
            model = models.Stokes2DCheat(
                config,
                u_inflow,
                inflow_coords,
                outflow_coords,
                wall_coords,
                cylinder_coords,
                Re,
                p_out,
                p_x_out,
                p_y_out,
            )

    # Initialize evaluator
    evaluator = models.StokesEvaluator(config, model)

    # Initialize residual sampler
    if config.cheat:
        res_sampler = iter(SpaceSamplerCheat(coords, (p_ref,p_x_ref,p_y_ref), config.training.batch_size_per_device))
    else:   
        res_sampler = iter(SpaceSampler(coords, config.training.batch_size_per_device))

    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        #print("Step: ", step)
        batch = next(res_sampler)
        #print("Batch: ", batch)
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                # if config.cheat:
                #     batch,_,_,_ = batch
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, coords, u_ref, v_ref)
                wandb.log(log_dict, step)

                end_time = time.time()
                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = time.time()

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model
