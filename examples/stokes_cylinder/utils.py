import jax.numpy as jnp

import scipy.io


# Inflow boundary condition
def parabolic_inflow(y, U_max):
    u = 4 * U_max * y * (0.41 - y) / (0.41**2)
    v = jnp.zeros_like(y)
    return u, v


def get_dataset(cheat=False):
    if cheat:
        data = jnp.load("data/stokes2.npy", allow_pickle=True).item()
    else:
        data = jnp.load("data/stokes.npy", allow_pickle=True).item()
    u_ref = jnp.array(data["u"])
    v_ref = jnp.array(data["v"])
    p_ref = jnp.array(data["p"])
    if cheat:
        p_x_ref = jnp.array(data["p_x"])
        p_y_ref = jnp.array(data["p_y"])
        p_x_out = jnp.array(data["p_x_out"])
        p_y_out = jnp.array(data["p_y_out"])
        p_out = jnp.array(data['p_out'])
        p_ref = (p_ref, p_x_ref, p_y_ref, p_out,p_x_out, p_y_out)
    coords = jnp.array(data["coords"])
    inflow_coords = jnp.array(data["inflow_coords"])
    outflow_coords = jnp.array(data["outflow_coords"])
    wall_coords = jnp.array(data["wall_coords"])
    cylinder_coords = jnp.array(data["cylinder_coords"])
    nu = jnp.array(data["nu"])
    return (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,
    )
