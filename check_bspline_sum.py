import jax
import jax.numpy as jnp

from jaxpi.archi_pyramid import cubic_bspline_weight


def check_partition_of_unity(res=64, num=10000, seed=0):
    key = jax.random.PRNGKey(seed)
    key_base, key_frac = jax.random.split(key)

    # Interior base_idx in [1, res-3] so all offsets [-1,0,1,2] stay in-bounds
    base = jax.random.randint(key_base, (num,), 1, res - 2)
    frac = jax.random.uniform(key_frac, (num,))
    gx = base + frac

    offsets = jnp.arange(-1, 3)
    idx = base[:, None] + offsets[None, :]
    dist = gx[:, None] - idx

    w = cubic_bspline_weight(dist)
    sum_w = jnp.sum(w, axis=1)

    return {
        "min": float(jnp.min(sum_w)),
        "max": float(jnp.max(sum_w)),
        "mean": float(jnp.mean(sum_w)),
        "max_abs_err": float(jnp.max(jnp.abs(sum_w - 1.0))),
    }


if __name__ == "__main__":
    stats = check_partition_of_unity()
    print("Partition-of-unity check (interior points):")
    for k, v in stats.items():
        print(f"{k}: {v:.6e}")
