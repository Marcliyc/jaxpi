from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn


class Stokes2D(ForwardBVP):
    def __init__(
        self,
        config,
        u_inflow,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        Re,
    ):
        super().__init__(config)

        self.u_in = u_inflow  # inflow profile
        self.Re = Re  # Reynolds number

        # Initialize coordinates
        self.inflow_coords = inflow_coords
        self.outflow_coords = outflow_coords
        self.wall_coords = wall_coords
        self.cylinder_coords = cylinder_coords
        self.noslip_coords = jnp.vstack((self.wall_coords, self.cylinder_coords))

        # Non-dimensionalized domain length and width
        self.L, self.W = self.noslip_coords.max(axis=0) - self.noslip_coords.min(axis=0)

        # Predict functions over batch
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))

    def neural_net(self, params, x, y):
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
        z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        return u, v, p

    def u_net(self, params, x, y):
        u, _, _ = self.neural_net(params, x, y)
        return u

    def v_net(self, params, x, y):
        _, v, _ = self.neural_net(params, x, y)
        return v

    def p_net(self, params, x, y):
        _, _, p = self.neural_net(params, x, y)
        return p

    def r_net(self, params, x, y):
        u, v, p = self.neural_net(params, x, y)

        (u_x, u_y), (v_x, v_y), (p_x, p_y) = jacrev(self.neural_net, argnums=(1, 2))(params, x, y)

        u_hessian = hessian(self.u_net, argnums=(1, 2))(params, x, y)
        v_hessian = hessian(self.v_net, argnums=(1, 2))(params, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        ru = -(u_xx + u_yy) + p_x
        rv = -(v_xx + v_yy) + p_y
        rc = u_x + v_y

        u_out = u_x - p
        v_out = v_x

        return ru, rv, rc, u_out, v_out

    def ru_net(self, params, x, y):
        ru, _, _, _, _ = self.r_net(params, x, y)
        return ru

    def rv_net(self, params, x, y):
        _, rv, _, _, _ = self.r_net(params, x, y)
        return rv

    def rc_net(self, params, x, y):
        _, _, rc, _, _ = self.r_net(params, x, y)
        return rc

    def u_out_net(self, params, x, y):
        _, _, _, u_out, _ = self.r_net(params, x, y)
        return u_out

    def v_out_net(self, params, x, y):
        _, _, _, _, v_out = self.r_net(params, x, y)
        return v_out

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Inflow boundary conditions
        u_in_pred = self.u_pred_fn(
            params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )
        v_in_pred = self.v_pred_fn(
            params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )

        u_in_loss = jnp.mean((u_in_pred - self.u_in) ** 2)
        v_in_loss = jnp.mean(v_in_pred**2)

        # Outflow boundary conditions
        _, _, _, u_out_pred, v_out_pred = self.r_pred_fn(
            params, self.outflow_coords[:, 0], self.outflow_coords[:, 1]
        )

        u_out_loss = jnp.mean(u_out_pred**2)
        v_out_loss = jnp.mean(v_out_pred**2)

        # No-slip boundary conditions
        u_noslip_pred = self.u_pred_fn(
            params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )
        v_noslip_pred = self.v_pred_fn(
            params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )

        u_noslip_loss = jnp.mean(u_noslip_pred**2)
        v_noslip_loss = jnp.mean(v_noslip_pred**2)

        # Residual losses
        ru_pred, rv_pred, rc_pred, _, _ = self.r_pred_fn(
            params, batch[:, 0], batch[:, 1]
        )

        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "u_out": u_out_loss,
            "v_out": v_out_loss,
            "u_noslip": u_noslip_loss,
            "v_noslip": v_noslip_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        u_in_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_net, params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )
        v_in_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_net, params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )

        u_out_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_out_net, params, self.outflow_coords[:, 0], self.outflow_coords[:, 1]
        )
        v_out_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_out_net, params, self.outflow_coords[:, 0], self.outflow_coords[:, 1]
        )

        u_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_net, params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )
        v_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_net, params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )

        ru_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.ru_net, params, batch[:, 0], batch[:, 1]
        )
        rv_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.rv_net, params, batch[:, 0], batch[:, 1]
        )
        rc_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.rc_net, params, batch[:, 0], batch[:, 1]
        )

        ntk_dict = {
            "u_in": u_in_ntk,
            "v_in": v_in_ntk,
            "u_out": u_out_ntk,
            "v_out": v_out_ntk,
            "u_noslip": u_noslip_ntk,
            "v_noslip": v_noslip_ntk,
            "ru": ru_ntk,
            "rv": rv_ntk,
            "rc": rc_ntk,
        }

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords, u_test, v_test):
        U_test = jnp.sqrt(u_test**2 + v_test**2)

        u_pred = self.u_pred_fn(params, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, coords[:, 0], coords[:, 1])
        U_pred = jnp.sqrt(u_pred**2 + v_pred**2)

        u_error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        v_error = jnp.linalg.norm(v_pred - v_test) / jnp.linalg.norm(v_test)
        U_error = jnp.linalg.norm(U_pred - U_test) / jnp.linalg.norm(U_test)

        return u_error, v_error, U_error

class Stokes2DCheat(ForwardBVP):
    def __init__(
        self,
        config,
        u_inflow,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        Re,
        p_out, # added to cheat (use p_ref)
        p_x_out,
        p_y_out
    ):
        super().__init__(config)

        self.u_in = u_inflow  # inflow profile
        self.Re = Re  # Reynolds number

        # Initialize coordinates
        self.inflow_coords = inflow_coords
        self.outflow_coords = outflow_coords
        self.wall_coords = wall_coords
        self.cylinder_coords = cylinder_coords
        self.noslip_coords = jnp.vstack((self.wall_coords, self.cylinder_coords))

        self.p_out = p_out
        self.p_x_out = p_x_out
        self.p_y_out = p_y_out

        # Non-dimensionalized domain length and width
        self.L, self.W = self.noslip_coords.max(axis=0) - self.noslip_coords.min(axis=0)

        # Predict functions over batch
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        #self.p_pred_fn = vmap(self.p_net, (None, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, 0, 0))

    def neural_net(self, params, x, y):
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
        z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        return u, v

    def u_net(self, params, x, y):
        u, _ = self.neural_net(params, x, y)
        return u

    def v_net(self, params, x, y):
        _, v = self.neural_net(params, x, y)
        return v

    def r_net(self, params, x, y, p, p_x, p_y): #p, p_x, p_y are ref
        u, v = self.neural_net(params, x, y)

        (u_x, u_y), (v_x, v_y) = jacrev(self.neural_net, argnums=(1, 2))(params, x, y)

        u_hessian = hessian(self.u_net, argnums=(1, 2))(params, x, y)
        v_hessian = hessian(self.v_net, argnums=(1, 2))(params, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        ru = -(u_xx + u_yy) + p_x
        rv = -(v_xx + v_yy) + p_y
        rc = u_x + v_y

        u_out = u_x - p
        v_out = v_x

        return ru, rv, rc, u_out, v_out

    def ru_net(self, params, x, y, p, p_x, p_y):
        ru, _, _, _, _ = self.r_net(params, x, y, p, p_x, p_y)
        return ru

    def rv_net(self, params, x, y, p, p_x, p_y):
        _, rv, _, _, _ = self.r_net(params, x, y, p, p_x, p_y)
        return rv

    def rc_net(self, params, x, y, p, p_x, p_y):
        _, _, rc, _, _ = self.r_net(params, x, y, p, p_x, p_y)
        return rc

    def u_out_net(self, params, x, y, p, p_x, p_y):
        _, _, _, u_out, _ = self.r_net(params, x, y, p, p_x, p_y)
        return u_out

    def v_out_net(self, params, x, y, p, p_x, p_y):
        _, _, _, _, v_out = self.r_net(params, x, y, p, p_x, p_y)
        return v_out

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Inflow boundary conditions
        batch, p, p_x, p_y = batch
        u_in_pred = self.u_pred_fn(
            params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )
        v_in_pred = self.v_pred_fn(
            params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )

        u_in_loss = jnp.mean((u_in_pred - self.u_in) ** 2)
        v_in_loss = jnp.mean(v_in_pred**2)

        # Outflow boundary conditions
        _, _, _, u_out_pred, v_out_pred = self.r_pred_fn(
            params, self.outflow_coords[:, 0], self.outflow_coords[:, 1], self.p_out, self.p_x_out, self.p_y_out
        )

        u_out_loss = jnp.mean(u_out_pred**2)
        v_out_loss = jnp.mean(v_out_pred**2)

        # No-slip boundary conditions
        u_noslip_pred = self.u_pred_fn(
            params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )
        v_noslip_pred = self.v_pred_fn(
            params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )

        u_noslip_loss = jnp.mean(u_noslip_pred**2)
        v_noslip_loss = jnp.mean(v_noslip_pred**2)


        # Residual losses
        ru_pred, rv_pred, rc_pred, _, _ = self.r_pred_fn(
            params, batch[:, 0], batch[:, 1], p, p_x, p_y
        )

        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "u_out": u_out_loss,
            "v_out": v_out_loss,
            "u_noslip": u_noslip_loss,
            "v_noslip": v_noslip_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        return 0

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords, u_test, v_test):
        U_test = jnp.sqrt(u_test**2 + v_test**2)

        u_pred = self.u_pred_fn(params, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, coords[:, 0], coords[:, 1])
        U_pred = jnp.sqrt(u_pred**2 + v_pred**2)

        u_error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        v_error = jnp.linalg.norm(v_pred - v_test) / jnp.linalg.norm(v_test)
        U_error = jnp.linalg.norm(U_pred - U_test) / jnp.linalg.norm(U_test)

        return u_error, v_error, U_error

class Stokes2DHardBCCheat(Stokes2DCheat):
    def __init__(
        self,
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
        cylinder_center=[2, 2], # depends on the data generation process
        cylinder_radius=0.5, # depends on the data generation process
        cylinder_dist = False,
        wall_dist = True,
    ):   
        super().__init__(config,u_inflow,inflow_coords,outflow_coords,wall_coords,cylinder_coords,Re,p_out,p_x_out,p_y_out)

        # --- Geometry Setup ---
        self.xc = cylinder_center[0]
        self.yc = cylinder_center[1]
        self.R = cylinder_radius
        self.cylinder_dist = cylinder_dist
        self.wall_dist = wall_dist

        # Detect Wall Boundaries from the provided coordinates
        self.y_min = wall_coords[:, 1].min()
        self.y_max = wall_coords[:, 1].max()
        
        self.x_min = wall_coords[:, 0].min()
        self.x_max = wall_coords[:, 0].max()
        self.L = self.x_max - self.x_min
        self.W = self.y_max - self.y_min

    def geometry_phi(self, x, y):
        """
        Returns a distance function that vanishes at:
        1. The Cylinder Surface
        2. The Top Wall
        3. The Bottom Wall
        """
        # A. Cylinder Distance (Algebraic)
        # Normalized by L^2 to keep values manageable (~order 1)
        phi = 1.0
        if self.cylinder_dist:
            if self.config.cyl_alpha > 0:
                phi_cyl = ((x - self.xc)**2 + (y - self.yc)**2 - self.R**2) / (self.R*2)#(self.R**2)
                alpha = self.config.cyl_alpha
                phi_cyl = jnp.tanh(alpha * phi_cyl) # smooth
            else:
                phi_cyl = ((x - self.xc)**2 + (y - self.yc)**2 - self.R**2) / (self.R*2)
            phi = phi * phi_cyl

        # B. Wall Distance (Parabolic profile vanishing at y_min and y_max)
        # Normalized by W^2
        if self.wall_dist:
            phi_wall = ((y - self.y_min) * (self.y_max - y)) / (self.W**2)
            phi = phi * phi_wall

        # C. Combine with R-Function if both distances are used
        if self.cylinder_dist and self.wall_dist:
            m = self.config.get("phi_m", 2.0)
            phi = phi_cyl*phi_wall*(phi_cyl**m+phi_wall**m)**(-1/m)

        return phi
    
    def neural_net(self, params, x, y):
        x_norm = x / self.L  # rescale x into [0, 1]
        y_norm = y / self.W  # rescale y into [0, 1]
        z = jnp.stack([x_norm, y_norm])
        outputs = self.state.apply_fn(params, z)
        phi = self.geometry_phi(x, y)
        u = outputs[0]*phi
        v = outputs[1]*phi
        return u, v

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # 1. Inflow Loss (Dirichlet, Soft)
        batch, p, p_x, p_y = batch

        u_in_pred = self.u_pred_fn(params, self.inflow_coords[:, 0], self.inflow_coords[:, 1])
        v_in_pred = self.v_pred_fn(params, self.inflow_coords[:, 0], self.inflow_coords[:, 1])
        u_in_loss = jnp.mean((u_in_pred - self.u_in) ** 2)
        v_in_loss = jnp.mean(v_in_pred**2)

        # 2. Outflow Loss (Neumann, Soft)
        _, _, _, u_out_pred, v_out_pred = self.r_pred_fn(
            params, self.outflow_coords[:, 0], self.outflow_coords[:, 1], self.p_out, self.p_x_out, self.p_y_out
        )
        u_out_loss = jnp.mean(u_out_pred**2)
        v_out_loss = jnp.mean(v_out_pred**2)

        # 3. PDE Residuals
        ru_pred, rv_pred, rc_pred, _, _ = self.r_pred_fn(params, batch[:, 0], batch[:, 1], p, p_x, p_y)
        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)
        
        # --- REMOVED: u_noslip_loss and v_noslip_loss ---
        # They are zero by definition.

        loss_dict = {
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "u_out": u_out_loss,
            "v_out": v_out_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }

        # Wall and cylinder losses if not using distance functions
        if not self.wall_dist:
            u_wall_pred = self.u_pred_fn(
                params, self.wall_coords[:, 0], self.wall_coords[:, 1]
            )
            v_wall_pred = self.v_pred_fn(
                params, self.wall_coords[:, 0], self.wall_coords[:, 1]
            )
            u_wall_loss = jnp.mean(u_wall_pred**2)
            v_wall_loss = jnp.mean(v_wall_pred**2)
            loss_dict["u_wall"] = u_wall_loss
            loss_dict["v_wall"] = v_wall_loss
        if not self.cylinder_dist:
            u_cyl_pred = self.u_pred_fn(
                params, self.cylinder_coords[:, 0], self.cylinder_coords[:, 1]
            )
            v_cyl_pred = self.v_pred_fn(
                params, self.cylinder_coords[:, 0], self.cylinder_coords[:, 1]
            )
            u_cyl_loss = jnp.mean(u_cyl_pred**2)
            v_cyl_loss = jnp.mean(v_cyl_pred**2)
            loss_dict["u_cyl"] = u_cyl_loss
            loss_dict["v_cyl"] = v_cyl_loss

        return loss_dict

class Stokes2DHardBC(Stokes2D):
    def __init__(
        self,
        config,
        u_inflow,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        Re,
        cylinder_center=[2, 2],
        cylinder_radius=0.5,
        cylinder_dist = False,
        wall_dist = True,
    ):   
        super().__init__(config,u_inflow,inflow_coords,outflow_coords,wall_coords,cylinder_coords,Re)

        # --- Geometry Setup ---
        #self.xc = cylinder_center[0]
        #self.yc = cylinder_center[1]
        self.xc, self.yc = cylinder_coords.mean(axis=0)
        #self.R = cylinder_radius
        self.R = jnp.sqrt(jnp.mean((cylinder_coords[:,0]-self.xc)**2 + (cylinder_coords[:,1]-self.yc)**2))
        self.cylinder_dist = cylinder_dist
        self.wall_dist = wall_dist

        # Detect Wall Boundaries from the provided coordinates
        self.y_min = wall_coords[:, 1].min()
        self.y_max = wall_coords[:, 1].max()
        
        self.x_min = wall_coords[:, 0].min()
        self.x_max = wall_coords[:, 0].max()
        self.L = self.x_max - self.x_min
        self.W = self.y_max - self.y_min

    def geometry_phi(self, x, y):
        """
        Returns a distance function that vanishes at:
        1. The Cylinder Surface
        2. The Top Wall
        3. The Bottom Wall
        """
        # A. Cylinder Distance (Algebraic)
        # Normalized by L^2 to keep values manageable (~order 1)
        phi = 1.0
        if self.cylinder_dist:
            if self.config.cyl_alpha > 0:
                phi_cyl = ((x - self.xc)**2 + (y - self.yc)**2 - self.R**2) / (self.R*2)#(self.R**2)
                alpha = self.config.cyl_alpha
                phi_cyl = jnp.tanh(alpha * phi_cyl)
            else:
                phi_cyl = ((x - self.xc)**2 + (y - self.yc)**2 - self.R**2) / (self.R*2)
            phi = phi * phi_cyl

        # B. Wall Distance (Parabolic profile vanishing at y_min and y_max)
        # Normalized by W^2
        if self.wall_dist:
            phi_wall = ((y - self.y_min) * (self.y_max - y)) / (self.W**2)
            phi = phi * phi_wall

        if self.cylinder_dist and self.wall_dist:
            m = self.config.get("phi_m", 2.0)
            phi = phi_cyl*phi_wall*(phi_cyl**m+phi_wall**m)**(-1/m)
        # C. Combine
        return phi# * phi_wall
    
    def neural_net(self, params, x, y):
        x_norm = x / self.L  # rescale x into [0, 1]
        y_norm = y / self.W  # rescale y into [0, 1]
        z = jnp.stack([x_norm, y_norm])
        outputs = self.state.apply_fn(params, z)
        phi = self.geometry_phi(x, y)
        #part = self.u_particular(x, y)
        u = outputs[0]*phi # + part
        v = outputs[1]*phi
        p = outputs[2]
        return u, v, p

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # 1. Inflow Loss (Dirichlet, Soft)
        # Note: Even though we have hard wall constraints, the Inflow at x=0
        # is distinct. The phi function is non-zero at x=0 (mostly), 
        # so the network can learn the inflow parabola.
        u_in_pred = self.u_pred_fn(params, self.inflow_coords[:, 0], self.inflow_coords[:, 1])
        v_in_pred = self.v_pred_fn(params, self.inflow_coords[:, 0], self.inflow_coords[:, 1])
        u_in_loss = jnp.mean((u_in_pred - self.u_in) ** 2)
        v_in_loss = jnp.mean(v_in_pred**2)

        # 2. Outflow Loss (Neumann, Soft)
        _, _, _, u_out_pred, v_out_pred = self.r_pred_fn(
            params, self.outflow_coords[:, 0], self.outflow_coords[:, 1]
        )
        u_out_loss = jnp.mean(u_out_pred**2)
        v_out_loss = jnp.mean(v_out_pred**2)

        # 3. PDE Residuals
        ru_pred, rv_pred, rc_pred, _, _ = self.r_pred_fn(params, batch[:, 0], batch[:, 1])
        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)
        
        # --- REMOVED: u_noslip_loss and v_noslip_loss ---
        # They are zero by definition.

        loss_dict = {
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "u_out": u_out_loss,
            "v_out": v_out_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }
        if not self.wall_dist:
            u_wall_pred = self.u_pred_fn(
                params, self.wall_coords[:, 0], self.wall_coords[:, 1]
            )
            v_wall_pred = self.v_pred_fn(
                params, self.wall_coords[:, 0], self.wall_coords[:, 1]
            )
            u_wall_loss = jnp.mean(u_wall_pred**2)
            v_wall_loss = jnp.mean(v_wall_pred**2)
            loss_dict["u_wall"] = u_wall_loss
            loss_dict["v_wall"] = v_wall_loss
        if not self.cylinder_dist:
            u_cyl_pred = self.u_pred_fn(
                params, self.cylinder_coords[:, 0], self.cylinder_coords[:, 1]
            )
            v_cyl_pred = self.v_pred_fn(
                params, self.cylinder_coords[:, 0], self.cylinder_coords[:, 1]
            )
            u_cyl_loss = jnp.mean(u_cyl_pred**2)
            v_cyl_loss = jnp.mean(v_cyl_pred**2)
            loss_dict["u_cyl"] = u_cyl_loss
            loss_dict["v_cyl"] = v_cyl_loss

        return loss_dict
    
class Stokes2DHardAll(Stokes2D):
    def __init__(
        self,
        config,
        u_inflow, # Scalar max velocity (e.g. 1.5) instead of array
        inflow_coords, outflow_coords, wall_coords, # Kept for domain sizing
        cylinder_coords,
        Re,
        cylinder_center=[2, 2],
        cylinder_radius=0.5,
        hard_outflow=False # Toggle to hard-constrain outlet to same profile as inlet
    ):
        super().__init__(config,u_inflow,inflow_coords,outflow_coords,wall_coords,cylinder_coords,Re)

        
        self.hard_outflow = hard_outflow
        self.u_max = 0.3#u_inflow_max

        # Geometry
        # self.L = 2.2
        # self.W = 0.41
        self.xc = cylinder_center[0]
        self.yc = cylinder_center[1]
        self.R = cylinder_radius

        # Boundaries
        self.x_min, self.x_max = 0.0, 2.2
        self.y_min, self.y_max = 0.0, 0.41

    # ==========================================================
    # 1. Particular Solution (The "Interpolation" Function)
    # ==========================================================
    def u_particular(self, x, y):
        """
        Constructs a particular solution using Algebraic Shielding.
        
        1. Background: Poiseuille Flow (Matches Walls & Inflow)
        2. Shielding:   (1 - R^2 / r^2) 
           - Derived from Potential Flow theory.
           - Smoothly zeros out velocity at cylinder surface.
           - Decays naturally as 1/r^2.
        """
        # 1. Background Flow (Parabolic Poiseuille)
        # u_channel = 4 * U_max * y * (W - y) / W^2
        u_channel = 4.0 * self.u_max * (y * (self.W - y)) / (self.W**2)

        # 2. Compute Squared Euclidean Distance from Center
        # r^2 = (x - xc)^2 + (y - yc)^2
        r_sq = (x - self.xc)**2 + (y - self.yc)**2
        
        # 3. Algebraic Shielding
        # We clamp r_sq to be at least R^2 to avoid division by zero 
        # inside the cylinder (though we only solve outside, this is safe).
        safe_r_sq = jnp.maximum(r_sq, self.R**2)
        
        # Factor = 1 - (R^2 / r^2)
        # This creates a "hole" in the flow exactly the size of the cylinder
        shield = 1.0 - (self.R**2 / safe_r_sq)

        # 4. Combine
        return u_channel * shield

    # ==========================================================
    # 2. Distance / Filter Function (The "Zero" Enforcer)
    # ==========================================================
    def filter_phi(self, x, y):
        """
        Must be zero on ALL Hard boundaries:
        - Cylinder
        - Walls
        - Inflow (x=0)
        - Outflow (x=L) [Only if hard_outflow=True]
        """
        # A. Cylinder & Walls (Same as before)
        phi_cyl = ((x - self.xc)**2 + (y - self.yc)**2 - self.R**2) / (self.R*2)
        alpha = self.config.cyl_alpha
        phi_cyl = jnp.tanh(alpha * phi_cyl)
        phi_wall = (y * (self.W - y)) / (self.W**2)

        # B. Inflow Constraint (x=0)
        # Term 'x' vanishes at x=0. Normalized by L.
        phi_in = x / self.L

        m = self.config.get("phi_m", 2.0)
        # C. Outflow Constraint (x=L)
        if self.hard_outflow:
            phi_out = (self.L - x) / self.L
            phi = (phi_cyl**(-m)+phi_wall**(-m)+phi_in**(-m)+phi_out**(-m))**(-1/m) 
        else:
            phi_out = 1.0 # No constraint at outlet
            phi = (phi_cyl**(-m)+phi_wall**(-m)+phi_in**(-m))**(-1/m) 

        return phi#phi_cyl * phi_wall * phi_in * phi_out

    # ==========================================================
    # 3. Neural Network Ansatz
    # ==========================================================
    def neural_net(self, params, x, y):
        x_norm = x / self.L  # rescale x into [0, 1]
        y_norm = y / self.W  # rescale y into [0, 1]
        z = jnp.stack([x_norm, y_norm])
        outputs = self.state.apply_fn(params, z)
        phi = self.filter_phi(x, y)
        part = self.u_particular(x, y)
        u = outputs[0]*phi + part
        v = outputs[1]*phi
        p = outputs[2]
        return u, v, p

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        ru_pred, rv_pred, rc_pred, _, _ = self.r_pred_fn(
            params, batch[:, 0], batch[:, 1]
        )

        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)
        
        loss_dict = {
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss
        }

        if not self.hard_outflow:
            _, _, _, u_stress_out, v_stress_out = self.r_pred_fn(
                params, self.outflow_coords[:,0], self.outflow_coords[:,1]
            )
            loss_dict["u_out"] = jnp.mean(u_stress_out**2)
            loss_dict["v_out"] = jnp.mean(v_stress_out**2)

        return loss_dict

class StokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, coords, u_ref, v_ref):
        u_error, v_error, U_error = self.model.compute_l2_error(
            params, coords, u_ref, v_ref
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["U_error"] = U_error

    # def log_preds(self, params, x_star, y_star):
    #     u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)
    #
    #     fig = plt.figure()
    #     plt.pcolor(U_pred.T, cmap='jet')
    #     log_dict['U_pred'] = fig
    #     fig.close()

    def __call__(self, state, batch, coords, u_ref, v_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, coords, u_ref, v_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params, coords)

        return self.log_dict
