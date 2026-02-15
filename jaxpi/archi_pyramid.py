from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict, Literal
from dataclasses import field

from flax import linen as nn
from flax.core.frozen_dict import freeze

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
        (4.0 - 6.0 * t**2 + 3.0 * t**3) / 6.0,
        jnp.where(t < 2.0, (2.0 - t)**3 / 6.0, 0.0)
    )


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
    feature_norm: Literal["none", "layer"] = "none"
    feature_scale: bool = True

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
            f = interpolate_grid_nd(
                grid,
                x,
                res=res,
                attn_mode=self.attn_mode,
                x_min=self.x_min,
                x_max=self.x_max,
            )
            if self.feature_norm == "layer":
                f = nn.LayerNorm(name=f"ln_{level}")(f)
            elif self.feature_norm != "none":
                raise ValueError(f"feature_norm '{self.feature_norm}' not supported")
            if self.feature_scale:
                scale = self.param(
                    f"feat_scale_{level}",
                    nn.initializers.ones,
                    (self.feature_dim,),
                )
                f = f * scale
            features.append(f)
            #features.append(interpolate_grid_2d(grid, x, x_min=self.x_min, x_max=self.x_max))
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
        else:
            raise ValueError("TimeDependentPINN requires a pyramid config; no features produced.")
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
        else:
            raise ValueError("TimeDependentPINN requires a pyramid config; no features produced.")
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
        else:
            raise ValueError("TimeDependentPINN requires a pyramid config; no features produced.")
        return TimeConditionedDecoder(
            hidden_mult=self.hidden_mult,
            time_embed_dim=self.time_embed_dim,
            max_period=self.max_period,
            out_dim=self.out_dim,
            activation=self.activation,
            reparam=self.reparam,
        ).decode_from(h, features, t, start_idx)
