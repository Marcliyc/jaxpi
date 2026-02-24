import ml_collections
import jax.numpy as jnp

def get_config():
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-AllenCahn"
    wandb.name = "gaussian2_muon"
    wandb.tag = None

    # Arch: Time-Dependent Multi-Res Grid
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PINN_Gaussian"
    arch.ndim = 2
    arch.grid_range = 2
    #arch.grid_shift = 1
    arch.num_gaussian = 4000
    arch.sigmas_range = 0.025
    arch.mlp_dim = 1
    arch.features = [16]
    arch.out_dim = 1
    arch.activation = "tanh"
    # arch.reparam = ml_collections.ConfigDict(
    #     {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    # )

    
    # # Domain boundaries for the grid (Must match your dataset)
    arch.x_min = jnp.array([0.0, -1.0])
    arch.x_max = jnp.array([1.0, 1.0])

    # Optim
    # config.optim = optim = ml_collections.ConfigDict()
    # optim.optimizer = "Adam"
    # optim.learning_rate = 1e-3
    # optim.decay_rate = 0.9         # S slightly slower decay helps grid methods
    # optim.decay_steps = 5000
    # optim.beta1 = 0.9
    # optim.beta2 = 0.999
    # optim.eps = 1e-8
    # optim.grad_accum_steps = 0
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Muon"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 5000
    optim.staircase = False
    optim.warmup_steps = 5000
    optim.grad_accum_steps = 0
    optim.schedule_free = False
    #optim.grad_clip = 5.0
    #optim.weight_decay=1e-5
    #optim.warmup_steps = 1000

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 300000     # Grid methods often converge faster than MLPs
    training.batch_size_per_device = 4096    # Large batch size is critical for grid-based methods

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = 'ntk'  # 'ntk' is heavy; grad_norm is a good balance
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1.0, "res": 1.0, "bc": 0.5, "bcx": 0.5}) # Emphasize ICs slightly
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

    config.input_dim = 2
    config.seed = 42
    config.bc_loss = True

    return config