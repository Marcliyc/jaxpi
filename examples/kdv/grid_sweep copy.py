import os

# os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For better reproducible!  ~35% slower !

from absl import app
from absl import flags

from ml_collections import config_flags

import wandb

import train

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    "./configs/grid_sweep.py",
    "File path to the training hyperparameter configuration.",
)


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    sweep_config = {
        "method": "grid",
        "name": "grid_sweep3",
        "metric": {"goal": "minimize", "name": "l2_error"},
    }

    parameters_dict = {
        "seed": {"values": [2]},
        "base_resolution": {"values": [128, 256]},
        "num_levels": {"values": [3,5]},
        "feat_dim": {"values": [48,96]},
        "time_embed_dim": {"values": [128, 256]},
        "max_period": {"values": [2.0]},
        "hidden_multi": {"values": [2,4]},
    }

    sweep_config["parameters"] = parameters_dict

    def train_sweep():
        config = FLAGS.config

        wandb.init(project=config.wandb.project, name=config.wandb.name)

        current = wandb.config

        # TimeDependentPINN grid settings
        config.arch.arch_name = "TimeDependentPINN"
        config.arch.pyramid.base_resolution = current.base_resolution
        config.arch.pyramid.num_levels = current.num_levels
        config.arch.pyramid.feature_dim = current.feat_dim
        config.arch.time_embed_dim = current.time_embed_dim
        config.arch.max_period = current.max_period
        config.arch.hidden_mult = current.hidden_multi

        # Reset PI-init each run to avoid stale least-squares coefficients.
        config.arch.pi_init = None
        config.seed = current.seed

        train.train_and_evaluate(config, workdir)

    sweep_id = wandb.sweep(
        sweep_config,
        project=config.wandb.project,
    )

    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
