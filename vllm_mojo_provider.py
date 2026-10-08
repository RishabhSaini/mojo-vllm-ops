"""Registers a Mojo kernel as a vLLM IR-op provider for `fused_add_rms_norm`.

This is the Phase 3 integration experiment: a Mojo kernel (compiled via
MAX's CustomOpLibrary) plugged into vLLM's IR-op dispatch system as the
"mojo" provider, alongside "vllm_c" (CUDA) and "native" (PyTorch).

Usage (before creating any vLLM config / running vLLM):
    import vllm_mojo_provider  # noqa: F401
"""

from pathlib import Path

import torch

import vllm.ir.ops as irops
from max.experimental.torch import CustomOpLibrary

_MOJO_OPS_DIR = Path(__file__).parent / "operations"
_ops = CustomOpLibrary(_MOJO_OPS_DIR)

# Each distinct epsilon compiles a separate kernel specialization (comptime
# parameter); cache them like a dispatch table.
_op_cache: dict[int, object] = {}


def _get_op(epsilon: float):
    micro = round(float(epsilon) * 1e6)
    if abs(float(epsilon) * 1e6 - micro) > 1e-9:
        return None
    if micro not in _op_cache:
        _op_cache[micro] = _ops.fused_add_rms_norm_mojo[{"epsilon_micro": micro}]
    return _op_cache[micro]


def _supports_args(x, x_residual, weight, epsilon, variance_size=None):
    if variance_size is not None:
        return False
    if weight is not None and weight.dtype != x.dtype:
        return False
    return _get_op(epsilon) is not None


@irops.fused_add_rms_norm.register_impl(
    "mojo",
    supported=torch.cuda.is_available(),
    supports_args=_supports_args,
    inplace=True,
)
def fused_add_rms_norm_mojo(
    x: torch.Tensor,
    x_residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mojo implementation of fused_add_rms_norm (in-place, like vllm_c).

    Call-site aliasing: the same tensors are passed as DPS outputs and
    inputs, so the kernel mutates x and x_residual in place with no copies.
    """
    _get_op(epsilon)(x, x_residual, x, x_residual, weight)
    return x, x_residual


print(
    "vllm_mojo_provider: registered 'mojo' impl for fused_add_rms_norm"
    f" (kernel library: {_MOJO_OPS_DIR})"
)
