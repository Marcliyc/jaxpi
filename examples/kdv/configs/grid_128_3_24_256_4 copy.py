import ml_collections
import jax.numpy as jnp


def get_config():
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-KDV"
    wandb.name = "grid_64_3_24_256_4"
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = False
    config.pi_init_type = "initial_condition"

    # Arch: Time-Dependent Multi-Res Grid
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "TimeDependentPINN"

    arch.pyramid = ml_collections.ConfigDict()
    arch.pyramid.num_levels = 3
    arch.pyramid.base_resolution = 128
    arch.pyramid.feature_dim = 24
    arch.pyramid.attn_mode = (1,)
    arch.pyramid.interp_method = "quantic"
    arch.pyramid.x_min = -1.0
    arch.pyramid.x_max = 1.0

    arch.time_embed_dim = 256
    arch.max_period = 2.0
    arch.hidden_mult = 4
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.pi_init = None
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 5000
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.grad_accum_steps = 0
    optim.warmup_steps = 0
    optim.staircase = False
    optim.schedule_free = False

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 200000
    training.batch_size_per_device = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 0.1
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = True
    logging.log_grads = True
    logging.log_ntk = True
    logging.log_nonlinearities = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 5

    config.input_dim = 1
    config.seed = 42

    return config
