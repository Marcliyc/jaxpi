from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict, Literal
from dataclasses import field


from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax
from jax import random, jit, vmap
from jax.nn import sigmoid
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant, uniform

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    trainable: Tuple[
        bool
    ]  # Specifies whether the period for each axis is trainable or not

    def setup(self):
        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]

        self.period_params = freeze(period_params)

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []

        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Embedding(nn.Module):
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        return x


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y


# TODO: Make it more general, e.g. imposing periodicity for the given axis

class FourierTimeEmbedding(nn.Module):
    embed_dim: int = 64
    max_period: float = 1.0
    activation: str = 'gelu'
    reparam: Union[None, Dict] = None
    
    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, t: float) -> jnp.ndarray:
        #$t = jnp.atleast_1d(t)[0]
        #half_dim = self.embed_dim // 2
        #freqs = jnp.exp(-jnp.log(self.max_period) * jnp.arange(half_dim) / half_dim)
        #args = t * freqs * 2 * jnp.pi
        #embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)])
        embedding = jnp.atleast_1d(t)
        embedding = Dense(features=self.embed_dim * 2, reparam=self.reparam)(embedding)
        embedding = self.activation_fn(embedding)
        return Dense(features=self.embed_dim, reparam=self.reparam)(embedding)


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class Bottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)

        x = (
            x + identity
        )  # Please note that the skip connection is added before the activation function, which is the same as the original ResNet

        x = self.activation_fn(x)

        return x


class PIBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        """
        Physics-informed bottleneck block: Add the skip connection after the activation function,
        which is different from the original ResNet, making it an identity mapping at initialization
        """
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        # alpha = jnp.exp(-alpha)

        x = alpha * x + (1 - alpha) * identity

        return x


class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, u, v):
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        x = alpha * x + (1 - alpha) * identity

        return x


class ResNet(nn.Module):
    arch_name: Optional[str] = "ResNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Bottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                reparam=self.reparam,
            )(x)

        y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class PIResNet(nn.Module):
    arch_name: Optional[str] = "PIResNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = PIBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
            )(x)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class PirateNet(nn.Module):
    arch_name: Optional[str] = "PirateNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        embs = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)
        x = embs

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        u = self.activation_fn(u)

        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
            )(x, u, v)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    pyramid: Union[None, Dict] = None
    gaussian: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        
        if self.gaussian:
            x = GaussianNd_Diag(**self.gaussian)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        if self.pyramid:
            x = SpatialFeaturePyramid(**self.pyramid)(x)
            x = jnp.concatenate(x, axis=-1)

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


#################################################################################################
#################################### neural operators ###########################################
#################################################################################################

class MlpBlock(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    activation: str
    reparam: Union[None, Dict]
    final_activation: bool

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        if self.final_activation:
            x = self.activation_fn(x)

        return x
    
# Multi-resolution grid
def cubic_bspline_weight(t):
    t = jnp.abs(t)
    return jnp.where(
        t < 1.0,
        2.0/3.0 - t**2 + 0.5*t**3,
        jnp.where(t < 2.0, (2.0 - t)**3 / 6.0, 0.0)
    )
# def cubic_bspline_weight(t):
#     t = jnp.abs(t)
#     return jnp.where(
#         t < 1.0,
#         2.0/3.0 - 1.5*t**2 + 0.5*t**3,
#         jnp.where(t < 2.0, (2.0 - t)**3 / 2.0, 0.0)
#     )

def linear_weight(t):
    return jnp.maximum(0.0, 1.0 - jnp.abs(t))

# In jaxpi/archs.py

def interpolate_grid_nd(grid, x, res=None,
                        attn_mode=(0,0), # Ensure this is a tuple as per previous fix
                        x_min=None, x_max=None, 
                        weight_fn=cubic_bspline_weight, offsets=jnp.arange(-1, 3)):
    ndim = len(attn_mode)
    
    if x_min is None:
        x_min = jnp.zeros((ndim,))
    if x_max is None:
        x_max = jnp.ones((ndim,))
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)
    domain_size = x_max - x_min
    x_norm = (x - x_min) / jnp.where(domain_size == 0, 1.0, domain_size)

    if res is None:
        res = min(grid.shape[:-1])
    N = jnp.array([res]*ndim)
    #print(N)
    gx = x_norm * (N)
    
    # Convert tuple to array for the 'add' calculation
    attn_mode_arr = jnp.array(attn_mode)
    add = jnp.where(attn_mode_arr == 0, 1, 0)
    gx += add
    
    base_idx = jnp.floor(gx)
    base_idx = base_idx.astype(jnp.int32)
    #print(base_idx)
    
    dim_weights = []
    dim_coords = []
    
    for d, mode in enumerate(attn_mode):
        idx_d = base_idx[d] + offsets 
        dist_d = gx[d] - idx_d
        w_d = weight_fn(dist_d)
        
        if mode == 0:  
            c_d = jnp.clip(idx_d, 0, N[d] - 1)
        else:          
            c_d = jnp.mod(idx_d, N[d])
            
        dim_weights.append(w_d)
        dim_coords.append(c_d)

    grid_patch = grid
    for axis, coords in enumerate(dim_coords):
        grid_patch = jnp.take(grid_patch, coords, axis=axis)

    combined_weights = dim_weights[0]
    for i in range(1, ndim):
        combined_weights = jnp.expand_dims(combined_weights, axis=-1)
        combined_weights = combined_weights * dim_weights[i]
    combined_weights = jnp.expand_dims(combined_weights, axis=-1)
    return jnp.sum(grid_patch * combined_weights, axis=tuple(range(ndim)))

class SpatialFeaturePyramid(nn.Module):
    num_levels: int = 6
    base_resolution: int = 4
    feature_dim: int = 48
    #ndim: int = 2                       # New: Support 1D, 2D, 3D
    attn_mode: tuple = (0,0) #0:'clip',1:'repeat', default ['clip','clip'] for 2D
    interp_method: str = 'cubic'        # New: 'linear' or 'cubic', not used currently
    x_min: Union[None, jnp.ndarray, float, int] = None
    x_max: Union[None, jnp.ndarray, float, int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> list:
        features = []
        # padding = 3
        for level in range(self.num_levels):
            res = self.base_resolution * (2 ** level)
            grid_shape = ()
            for mode in self.attn_mode:
                if mode == 0: #clip
                    padding = 3
                elif mode == 1: #repeat for periodic bc
                    padding = 0
                else:
                    raise NotImplementedError(f'Attention mode {mode} not supported!')
                grid_shape += (res + padding,)
            grid_shape += (self.feature_dim,)
            grid = self.param(
                f'grid_{level}',
                nn.initializers.normal(0.01),
                grid_shape
            )
            features.append(interpolate_grid_nd(grid, x, res=res, attn_mode=self.attn_mode, x_min=self.x_min, x_max=self.x_max))
            #features.append(interpolate_grid_2d(grid, x, x_min=self.x_min, x_max=self.x_max))
        return features[::-1]
    
# def normalize(x, x_min, x_max, to="minus1_1"):
#     x = (x - x_min) / (x_max - x_min)
#     if to == "minus1_1":
#         return 2.0 * x - 1.0
#     return x

class GaussianNd_Diag(nn.Module):
    ndim: int = 2
    num_gaussian: int = 100
    grid_range: float = 1.
    #grid_shift: Union[None, jnp.ndarray] = None
    sigmas_range: float = 0.5
    mlp_dim: int = 4
    x_min: Union[None, jnp.ndarray, float, int] = None
    x_max: Union[None, jnp.ndarray, float, int] = None

    def setup(self):
        # Parameters for N dimensions
        # mu: (mlp_dim, num_gaussian, ndim)
        self.mu = self.param("mu", uniform(self.grid_range), (self.mlp_dim, self.num_gaussian, self.ndim))
        # sigmas: (mlp_dim, num_gaussian, ndim)
        self.sigmas = self.param("sigmas", constant(self.sigmas_range), (self.mlp_dim, self.num_gaussian, self.ndim))
        self.weight = self.param("weight", normal(), (self.mlp_dim, self.num_gaussian, 1))

    @nn.compact
    def __call__(self, x):
        # if self.grid_shift:
        #     x = x+self.grid_shift
        x = (x - self.x_min) / (self.x_max - self.x_min) * self.grid_range

        x = jnp.atleast_1d(x)
        # x shape: (ndim,)
        # Broadcast x to (1, 1, ndim) to match params (mlp_dim, num_gaussian, ndim)
        diff = (x[None, None, :] - self.mu) / self.sigmas
        
        # Sum of squares along the spatial dimension (last axis)
        log_pdf = -0.5 * jnp.sum(diff**2, axis=-1)  # (mlp_dim, num_gaussian)
        pdf = jnp.exp(log_pdf)

        # weight is (mlp_dim, num_gaussian, 1), squeeze to match pdf
        w = self.weight[..., 0]

        # Weighted sum over gaussians (axis 1)
        output = jnp.sum(pdf * w, axis=1)  # (mlp_dim,)

        return output
    
class GaussianNd_Shared(nn.Module):
    ndim: int = 2
    num_gaussian: int = 100
    grid_range: float = 1.
    grid_shift: Union[None, jnp.ndarray] = None
    sigmas_range: float = 0.5
    mlp_dim: int = 4

    def setup(self):
        # Shared Parameters for N dimensions
        # mu: (num_gaussian, ndim)
        self.mu = self.param("mu", uniform(self.grid_range), (self.num_gaussian, self.ndim))
        # sigmas: (num_gaussian, ndim)
        self.sigmas = self.param("sigmas", constant(self.sigmas_range), (self.num_gaussian, self.ndim))

        # Mixing weights: (num_gaussian, mlp_dim)
        self.weight = self.param("weight", normal(), (self.num_gaussian, self.mlp_dim))

    @nn.compact
    def __call__(self, x):
        if self.grid_shift:
            x = x+self.grid_shift

        x = jnp.atleast_1d(x)
        # x shape: (ndim,)
        # Broadcast x to (1, ndim) to match params (num_gaussian, ndim)
        diff = (x[None, :] - self.mu) / self.sigmas

        # Sum of squares along the spatial dimension (last axis)
        log_pdf = -0.5 * jnp.sum(diff**2, axis=-1)  # (num_gaussian,)
        pdf = jnp.exp(log_pdf)

        # Weighted combination of the shared gaussians
        # (num_gaussian,) @ (num_gaussian, mlp_dim) -> (mlp_dim,)
        output = pdf @ self.weight

        return output
    
class PINN_Gaussian(nn.Module):
    features: Sequence[int]
    out_dim: int
    # pos_enc: int
    num_gaussian: int = 100
    grid_range: float = 2.
    #grid_shift: Union[None, jnp.ndarray] = None
    sigmas_range : float = 15.
    mlp_dim: int = 4
    ndim: int = 2
    activation: str = 'tanh'
    reparam: Union[None, Dict] = None
    arch_name: Optional[str] = "PINN_Gaussian"
    x_min: Union[None, jnp.ndarray, float, int] = None
    x_max: Union[None, jnp.ndarray, float, int] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        X = GaussianNd_Diag(ndim=self.ndim, num_gaussian=self.num_gaussian, grid_range=self.grid_range,  sigmas_range=self.sigmas_range, mlp_dim=self.mlp_dim, x_min=self.x_min, x_max=self.x_max)(x)
        # X = Gaussian3d_Full(self.num_gaussian, self.grid_range, self.sigmas_range, self.mlp_dim)(x,y,z)
        
        #init = nn.initializers.glorot_normal()
        for fs in self.features[:-1]:
            X = Dense(fs, reparam=self.reparam)(X)
            X = self.activation_fn(X)
        X = Dense(self.features[-1], reparam=self.reparam)(X)
        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return X, y

class SpatialFeatureGaussian(nn.Module):
    num_levels: int = 6
    base_resolution: int = 4
    feature_dim: int = 48
    ndim: int = 2
    grid_range: float = 1.
    grid_shift: Union[None, jnp.ndarray] = None
    shared: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> list:
        features = []
        for level in range(self.num_levels):
            res = self.base_resolution * (2 ** level)
            # Use res as number of gaussians per level
            # Pass the full vector x to the N-d module
            if self.shared:
                feat = GaussianNd_Shared(
                    ndim=self.ndim,
                    grid_range = self.grid_range,
                    grid_shift = self.grid_shift,
                    num_gaussian=res, 
                    sigmas_range=1/res, 
                    mlp_dim=self.feature_dim, 
                    name=f'gauss_{level}'
                )(x)
            else:
                feat = GaussianNd_Diag(
                    ndim=self.ndim,
                    grid_range = self.grid_range,
                    grid_shift = self.grid_shift,
                    num_gaussian=res, 
                    sigmas_range=1/res, 
                    mlp_dim=self.feature_dim, 
                    name=f'gauss_{level}'
                )(x)
            features.append(feat)
        return features[::-1]

class TimeConditionedGatedBlock(nn.Module):
    hidden_dim: int
    activation: str = "gelu"
    reparam: Union[None, Dict] = None
    
    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, h_fine: jnp.ndarray, f_coarse: jnp.ndarray, t_embed: jnp.ndarray) -> jnp.ndarray:
        dim = f_coarse.shape[-1]
        h_proj = Dense(features=dim, reparam=self.reparam, name='proj')(h_fine)
        # gate = jax.nn.sigmoid(self.param('gate', nn.initializers.zeros, (dim,)))
        gate = sigmoid(self.param('gate', nn.initializers.zeros, (dim,)))
        combined = f_coarse + gate * h_proj
        h = Dense(features=self.hidden_dim, reparam=self.reparam, name='ffn1')(combined)
        h = self.activation_fn(h)
        h = Dense(features=dim, reparam=self.reparam, name='ffn2')(h)
        h = combined + h
        
        # FiLM with small init for stability
        gamma = Dense(features=dim, kernel_init=nn.initializers.zeros, reparam=self.reparam,name='film_gamma')(t_embed)
        beta = Dense(features=dim, kernel_init=nn.initializers.zeros, reparam=self.reparam, name='film_beta')(t_embed)
        return (1 + 0.1 * gamma) * h + 0.1 * beta  # Scale down FiLM effect


class TimeConditionedDecoder(nn.Module):
    hidden_mult: int = 2
    time_embed_dim: int = 64
    max_period: float = 1.0
    out_dim: int = 1
    activation: str = "gelu"
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)
    
    @nn.compact
    def __call__(self, features: list, t: float, return_states: bool = False):
        L = len(features)
        dim = features[0].shape[-1]
        hidden = dim * self.hidden_mult

        t_embed = FourierTimeEmbedding(embed_dim=self.time_embed_dim, max_period=self.max_period, reparam=self.reparam)(t)

        h = features[0]
        states = [h] if return_states else None
        for k in range(L - 1):
            h = TimeConditionedGatedBlock(hidden_dim=hidden, name=f'fusion_{k}', activation=self.activation, reparam=self.reparam)(
                h, features[k + 1], t_embed
            )
            if return_states:
                states.append(h)
        
        h = jnp.concatenate([h, t_embed])
        h = Dense(features=128, name='head1', reparam=self.reparam)(h)
        h = self.activation_fn(h)
        h = Dense(features=64, name='head2', reparam=self.reparam)(h)
        h = self.activation_fn(h)
        out = Dense(features=self.out_dim, name='head_out', reparam=self.reparam)(h)#[0]
        if return_states:
            return out, states
        return out

    @nn.compact
    def decode_from(self, h: jnp.ndarray, features: list, t: float, start_idx: int) -> jnp.ndarray:
        L = len(features)
        dim = features[0].shape[-1]
        hidden = dim * self.hidden_mult

        t_embed = FourierTimeEmbedding(embed_dim=self.time_embed_dim, max_period=self.max_period, reparam=self.reparam)(t)

        for k in range(start_idx, L - 1):
            h = TimeConditionedGatedBlock(hidden_dim=hidden, name=f'fusion_{k}', activation=self.activation, reparam=self.reparam)(
                h, features[k + 1], t_embed
            )

        h = jnp.concatenate([h, t_embed])
        h = Dense(features=128, name='head1', reparam=self.reparam)(h)
        h = self.activation_fn(h)
        h = Dense(features=64, name='head2', reparam=self.reparam)(h)
        h = self.activation_fn(h)
        return Dense(features=self.out_dim, name='head_out', reparam=self.reparam)(h)
    
class TimeDependentPINN(nn.Module):
    arch_name: Optional[str] = "TimeDependentPINN"
    hidden_mult: int = 2
    time_embed_dim: int = 64
    max_period: float = 1.0
    out_dim: int = 1
    activation: str = 'gelu'
    reparam: Union[None, Dict] = None
    pyramid: Union[None, Dict] = None
    gaussian: Union[None, Dict] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        if self.pyramid:
            features = SpatialFeaturePyramid(**self.pyramid)(x)
        elif self.gaussian:
            features = SpatialFeatureGaussian(**self.gaussian)(x)
        u = TimeConditionedDecoder(
            hidden_mult=self.hidden_mult,
            time_embed_dim=self.time_embed_dim,
            max_period=self.max_period,
            out_dim=self.out_dim,
            activation=self.activation,
            reparam=self.reparam,
        )(features, t)
        #spatial_mask = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * 16.0
        return u #* spatial_mask

    @nn.compact
    def forward_with_states(self, x: jnp.ndarray, t: float):
        if self.pyramid:
            features = SpatialFeaturePyramid(**self.pyramid)(x)
        elif self.gaussian:
            features = SpatialFeatureGaussian(**self.gaussian)(x)
        u, states = TimeConditionedDecoder(
            hidden_mult=self.hidden_mult,
            time_embed_dim=self.time_embed_dim,
            max_period=self.max_period,
            out_dim=self.out_dim,
            activation=self.activation,
            reparam=self.reparam,
        )(features, t, return_states=True)
        return u, states

    @nn.compact
    def decode_from(self, x: jnp.ndarray, t: float, h: jnp.ndarray, start_idx: int) -> jnp.ndarray:
        if self.pyramid:
            features = SpatialFeaturePyramid(**self.pyramid)(x)
        elif self.gaussian:
            features = SpatialFeatureGaussian(**self.gaussian)(x)
        return TimeConditionedDecoder(
            hidden_mult=self.hidden_mult,
            time_embed_dim=self.time_embed_dim,
            max_period=self.max_period,
            out_dim=self.out_dim,
            activation=self.activation,
            reparam=self.reparam,
        ).decode_from(h, features, t, start_idx)

class DeepONet(nn.Module):
    arch_name: Optional[str] = "DeepONet"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x):
        u = MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            final_activation=False,
            reparam=self.reparam,
        )(u)

        x = Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)

        y = u * x
        y = self.activation_fn(y)
        y = Dense(features=self.out_dim, reparam=self.reparam)(y)
        return y
