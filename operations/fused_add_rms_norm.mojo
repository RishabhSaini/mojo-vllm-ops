# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# ===----------------------------------------------------------------------=== #
"""Fused residual-add + RMSNorm custom op for PyTorch, mirroring vLLM's
CUDA `fused_add_rms_norm` kernel (csrc/libtorch_stable/layernorm_kernels.cu).

In-place semantics (one thread-block per row), identical to vLLM:
    residual = x + residual             (bf16 add, like CUDA)
    variance = mean(residual^2)         (fp32 accumulation)
    x        = residual * rsqrt(variance + eps) * weight
"""

import extensibility

from extensibility import InputTensor, OutputTensor

from max.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext, get_gpu_target
import max.gpu.primitives.warp as warp
from max.gpu.sync import barrier
from std.math import ceildiv, rsqrt
from std.math.uutils import ufloordiv
from std.memory import unsafe_stack_allocation
from std.sys.info import simd_width_of, size_of
from std.utils.numerics import get_accum_type


comptime MAX_WARPS = 8


@inline(.always)
def _block_sum[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
    """Sums `val` across all threads of a block of at most MAX_WARPS warps."""
    var shared = unsafe_stack_allocation[
        MAX_WARPS, dtype, address_space=.SHARED
    ]()
    var broadcast = unsafe_stack_allocation[1, dtype, address_space=.SHARED]()

    var warp_sum = warp.sum(val)
    var wid = warp_id[broadcast=True]()
    var lane = lane_id()

    if lane == 0:
        shared[wid] = warp_sum
    barrier()

    if wid == 0:
        var block_sum = Scalar[dtype](0)
        if lane < ufloordiv(block_dim.x, WARP_SIZE):
            block_sum = shared[lane]
        block_sum = warp.lane_group_sum[num_lanes=MAX_WARPS](block_sum)
        if lane == 0:
            broadcast[0] = block_sum
    barrier()
    return broadcast[0]


def _fused_add_rms_norm_kernel[
    dtype: DType, width: Int, epsilon: Float64
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    res_out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    x_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    res_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    w_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    hidden: Int64,
):
    """In-place fused add + RMSNorm: out_ptr/res_out_ptr alias x_ptr/res_ptr
    in production, but are kept separate parameters so benchmarks can also
    measure out-of-place behavior."""
    comptime accum = get_accum_type[dtype]()
    comptime align = width * size_of[dtype]()

    var hid = Int(hidden)
    var tid = thread_idx.x
    var row = block_idx.x
    var base = row * hid

    var variance = Scalar[accum](0)

    # Pass 1: res_out = x + residual; accumulate sum of squares in fp32.
    for i in range(tid, ceildiv(hid, width), block_dim.x):
        var offset = base + i * width
        var x_vec = x_ptr.unsafe_load[width=width, alignment=align](offset)
        var r_vec = res_ptr.unsafe_load[width=width, alignment=align](offset)
        var temp = x_vec + r_vec
        res_out_ptr.unsafe_store[width, alignment=align](offset, temp)
        var temp_f = temp.cast[accum]()
        variance += (temp_f * temp_f).reduce_add()

    var total = _block_sum[accum](variance)
    var scale = rsqrt(
        total / Scalar[accum](hid) + Scalar[accum](epsilon)
    )

    # Pass 2: out = res_out * scale * weight.
    for i in range(tid, ceildiv(hid, width), block_dim.x):
        var offset = base + i * width
        var t_vec = res_out_ptr.unsafe_load[width=width, alignment=align](offset)
        var w_vec = w_ptr.unsafe_load[width=width, alignment=align](
            i * width
        )
        var o_vec = (
            t_vec.cast[accum]() * scale * w_vec.cast[accum]()
        ).cast[dtype]()
        out_ptr.unsafe_store[width, alignment=align](offset, o_vec)


@extensibility.register("fused_add_rms_norm_mojo")
struct FusedAddRMSNorm[epsilon_micro: Int]:
    """Fused residual-add + RMSNorm registered as a torch custom op.

    Destination-passing-style like the CustomOpLibrary bridge requires:
    `op(norm_out, out_residual, x, residual, weight)`. Pass the same tensors
    for (norm_out, x) and (out_residual, residual) to get vLLM's in-place
    semantics with zero extra copies.

    `epsilon_micro` is the RMSNorm epsilon in millionths (1 -> 1e-6), kept
    integral so the generated torch op schema name stays valid.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        norm_out: OutputTensor[dtype=dtype, rank=rank, ...],
        out_residual: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        residual: InputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert target == "gpu", "GPU-only prototype"
        comptime width = simd_width_of[dtype, target=get_gpu_target()]()

        var num_tokens = norm_out.dim_size(0)
        var hidden = norm_out.dim_size(1)
        assert hidden % width == 0, "hidden size must be a multiple of width"

        comptime kernel = _fused_add_rms_norm_kernel[
            dtype, width, Scalar[DType.float64](Self.epsilon_micro) * 1e-6
        ]

        # When outputs alias inputs (in-place mode), rebind once so the same
        # origin is not embedded in two arguments.
        var op_ptr = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
            norm_out.unsafe_ptr()
        )
        var orp = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
            out_residual.unsafe_ptr()
        )
        var xp = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](x.unsafe_ptr())
        var rp = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
            residual.unsafe_ptr()
        )
        ctx.enqueue_function[kernel](
            op_ptr,
            orp,
            xp,
            rp,
            weight.unsafe_ptr(),
            Int64(hidden),
            grid_dim=(num_tokens,),
            block_dim=(MAX_WARPS * WARP_SIZE,),
        )
