"""mamba module"""
import math

import torch
from einops import rearrange
from flash_attn.ops.triton.layer_norm import RMSNorm
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined, mamba_chunk_scan_combined
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    raise ImportError("Please install the mamba-ssm package to use the MambaMixer module.")

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    raise ImportError("Please install the causal-conv1d package to use the MambaMixer module.")

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)

class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_max = config.time_step_max
        self.time_step_min = config.time_step_min
        dt_init_floor = 1e-4
        self.dt_limit = (0.0, float("inf"))

        A_init_range = config.A_init_range

        self.head_dim = self.intermediate_size // config.ssm_num_head
        assert self.intermediate_size % config.ssm_num_head == 0, "hidden_size must be divisible by n_head"
        self.num_heads = config.ssm_num_head

        self.use_conv_bias = config.use_conv_bias

        self.chunk_size = 256

        conv_dim = self.intermediate_size + 2 * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=conv_dim,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        assert self.activation in ["silu"], "Only silu and swish activations are supported"
        self.act = nn.SiLU

        # projection of the input hidden states
        d_in_proj = 2 * self.intermediate_size + 2 * self.ssm_state_size + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, d_in_proj, bias=config.use_bias)
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(self.time_step_max) - math.log(self.time_step_min))
            + math.log(self.time_step_min)
        )

        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.num_heads, ))
        self.D_has_hdim = False
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        self.norm = RMSNormGated(
            self.intermediate_size,
            eps=1e-6,
            norm_before_gate=False,
            group_size=self.intermediate_size,
        )
        self.rmsnorm = True

    def cuda_kernels_forward(self, hidden_states: torch.Tensor):
        zxbcdt = self.in_proj(hidden_states)
        A = -torch.exp(self.A_log)
        seq_idx = None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.training:  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.head_dim,
                ngroups=1,
                norm_before_gate=False,
                **dt_limit_kwargs,
            )
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.intermediate_size - 2 * self.ssm_state_size - self.num_heads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [
                    d_mlp,
                    d_mlp,
                    self.intermediate_size,
                    self.intermediate_size + 2 * self.ssm_state_size,
                    self.num_heads],
                dim=-1
            )
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.intermediate_size, self.ssm_state_size, self.ssm_state_size], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.head_dim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=1),
                rearrange(C, "b l (g n) -> b l g n", g=1),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.head_dim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.head_dim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=False,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            contextualized_states = self.out_proj(y)
        return contextualized_states

    def forward(self, hidden_states):
        if is_fast_path_available:
            hidden_states = self.cuda_kernels_forward(hidden_states)
        else:
            raise NotImplementedError("The fast path is not available. Please install the required libraries.")
        return hidden_states


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm = RMSNorm(self.hidden_size, eps=1e-6)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.act = nn.SiLU()
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden_states)


class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = RMSNorm(config.hidden_size, eps=1e-6)

        self.mixer = MambaMixer(config)
        self.mixer_rev = MambaMixer(config)
        # self.mlp = MLP(config)

    def forward(self, hidden_states, gen_noise):
        bzs, _ , h_dim = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            original_dtype = hidden_states.dtype
            residual = residual.to(torch.float32)
            hidden_states = hidden_states.to(torch.float32)
            self.mixer.to(torch.float32)
            self.mixer_rev.to(torch.float32)

        out = self.mixer(hidden_states)
        if gen_noise:
            out_rev = self.mixer_rev(
                hidden_states.flip(dims=(1,))[..., 1:, :]
            ).flip(dims=(1,))
            hidden_states = out + torch.cat(
                [out_rev, torch.zeros([bzs, 1, h_dim]).to(device=out.device, dtype=out.dtype)],
                dim=1
            )
        else:
            out_rev = self.mixer_rev(hidden_states.flip(dims=(1,))).flip(dims=(1,))
            hidden_states = out + out_rev
        if self.residual_in_fp32:
            hidden_states = hidden_states.to(original_dtype)
            residual = residual.to(original_dtype)
        hidden_states = residual + hidden_states
        # residual = hidden_states
        # hidden_states = self.mlp(hidden_states)
        return hidden_states    # + residual
