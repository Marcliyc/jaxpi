from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict, Literal

from flax import linen as nn
from flax.core.frozen_dict import freeze

from jax import random, jit, vmap
from jax.nn import sigmoid
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant

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
        t = jnp.atleast_1d(t)[0]
        half_dim = self.embed_dim // 2
        freqs = jnp.exp(-jnp.log(self.max_period) * jnp.arange(half_dim) / half_dim)
        args = t * freqs * 2 * jnp.pi
        embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)])
        embedding = Dense(features=self.embed_dim * 2, reparam=self.reparam)(embedding)
        embedding = self.activation_fn(embedding)
        return Dense(features=self.embed_dim, reparam=self.reparam)(embedding)

class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    pyramid: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        # if self.periodicity:
        #     x = PeriodEmbs(**self.periodicity)(x)

        # if self.fourier_emb:
        #     x = FourierEmbs(**self.fourier_emb)(x)

        if self.pyramid:
            x = SpatialFeaturePyramid(**self.pyramid)(x)

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x


# Multi-resolution grid
def cubic_bspline_weight(t):
    t = jnp.abs(t)
    return jnp.where(
        t < 1.0,
        2.0/3.0 - t**2 + 0.5*t**3,
        jnp.where(t < 2.0, (2.0 - t)**3 / 6.0, 0.0)
    )

def linear_weight(t):
    return jnp.maximum(0.0, 1.0 - jnp.abs(t))


def interpolate_grid_nd(grid, x, 
                        attn_mode = jnp.array([0,0]), # default 'clip' for all dims
                        x_min=None, x_max=None, 
                        weight_fn=cubic_bspline_weight, offsets = jnp.arange(-1, 3)):
    ndim=len(attn_mode)

    # Scaling
    if x_min is None:
        x_min = jnp.zeros((ndim,))
    if x_max is None:
        x_max = jnp.ones((ndim,))
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)
    domain_size = x_max - x_min
    x_norm = (x - x_min) / domain_size#jnp.where(domain_size == 0, 1.0, domain_size)


    N = jnp.array([min(grid.shape[:-1])]*ndim)  # resolution
    gx = x_norm * (N)
    add = jnp.where(attn_mode == 0, 1, 0)  # clip:1, repeat:0
    base_idx = jnp.floor(gx)+add
    base_idx = base_idx.astype(jnp.int32)

    dim_weights = []
    dim_coords = []
    for d,mode in enumerate(attn_mode):
        idx_d = base_idx[d] + offsets # Shape: (Kernel size,)
        dist_d = gx[d] - idx_d
        w_d = weight_fn(dist_d)       # Shape: (K,)
        if mode == 0:  # clip
            c_d = jnp.clip(idx_d, 0, N[d] - 1)
        else:          # repeat
            c_d = jnp.mod(idx_d, N[d])
        dim_weights.append(w_d)
        dim_coords.append(c_d)

    # combine weights
    grid_patch = grid[jnp.ix_(*dim_coords)]
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
    attn_mode: jnp.ndarray = jnp.array([0,0]) #0:'clip',1:'repeat', default ['clip','clip'] for 2D
    interp_method: str = 'cubic'        # 'linear' or 'cubic'
    x_min: Union[None, jnp.ndarray, float, int] = None
    x_max: Union[None, jnp.ndarray, float, int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> list:
        features = []
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
            features.append(interpolate_grid_nd(grid, x, attn_mode=self.attn_mode, x_min=self.x_min, x_max=self.x_max, method=self.interp_method))
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
    def __call__(self, features: list, t: float) -> jnp.ndarray:
        L = len(features)
        dim = features[0].shape[-1]
        hidden = dim * self.hidden_mult

        t_embed = FourierTimeEmbedding(embed_dim=self.time_embed_dim, max_period=self.max_period, reparam=self.reparam)(t)

        h = features[0]
        for k in range(L - 1):
            h = TimeConditionedGatedBlock(hidden_dim=hidden, name=f'fusion_{k}', activation=self.activation, reparam=self.reparam)(
                h, features[k + 1], t_embed
            )
        
        h = jnp.concatenate([h, t_embed])
        h = Dense(features=128, name='head1', reparam=self.reparam)(h)
        h = self.activation_fn(h)
        h = Dense(features=64, name='head2', reparam=self.reparam)(h)
        h = self.activation_fn(h)
        return Dense(features=self.out_dim, name='head_out', reparam=self.reparam)(h)#[0]
    
class TimeDependentPINN(nn.Module):
    arch_name: Optional[str] = "TimeDependentPINN"
    num_levels: int = 6
    base_resolution: int = 4
    feature_dim: int = 48
    attn_mode: jnp.ndarray = jnp.array([0,0])
    interp_method: str = 'cubic'        # 'linear' or 'cubic'
    hidden_mult: int = 2
    time_embed_dim: int = 64
    max_period: float = 1.0
    out_dim: int = 1
    activation: str = 'gelu'
    reparam: Union[None, Dict] = None
    x_min: Union[None, jnp.ndarray, float, int] = None
    x_max: Union[None, jnp.ndarray, float, int] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        features = SpatialFeaturePyramid(
            num_levels=self.num_levels,
            base_resolution=self.base_resolution,
            feature_dim=self.feature_dim,
            attn_mode=self.attn_mode,
            interp_method=self.interp_method,
            x_min=self.x_min,
            x_max=self.x_max,
        )(x)
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