import ml_collections
import jax.numpy as jnp

def get_config():
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-AllenCahn"
    wandb.name = "grid_1+1D_tanh_reparam2"
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = False

    # Arch: Time-Dependent Multi-Res Grid
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "TimeDependentPINN"
    arch.num_levels = 6             # 4-5 levels is usually sufficient for 1D AC; 7 might be overkill/slower
    arch.base_resolution = 4        # Start slightly coarser
    arch.feature_dim = 64           # 256 is very heavy for 1D; 64 or 128 is usually enough
    arch.attn_mode = (1,) # [1] = Periodic BCs (Repeat)
    arch.interp_method = 'cubic'
    arch.time_embed_dim = 256
    arch.max_period = 2.0
    arch.hidden_mult = 4            # Kept the second value from your file
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.pi_init = None
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )
    
    # Domain boundaries for the grid (Must match your dataset)
    arch.x_min = -1.0
    arch.x_max = 1.0

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9         # S slightly slower decay helps grid methods
    optim.decay_steps = 5000
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 300000     # Grid methods often converge faster than MLPs
    training.batch_size_per_device = 8192

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = 'ntk'  # 'ntk' is heavy; grad_norm is a good balance
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1.0, "res": 1.0}) # Emphasize ICs slightly
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    # Causal Training (CRITICAL for Allen-Cahn)
    # Even with a better arch, the stiffness of AC requires respecting time causality.
    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = True        # Turn this ON to see if the grid is learning physically valid sol.
    logging.log_grads = True
    logging.log_ntk = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 3

    # CRITICAL FIX: Spatial dimension only
    config.input_dim = 1
    config.seed = 42

    return config