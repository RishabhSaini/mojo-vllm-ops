# ===----------------------------------------------------------------------=== #
# Standalone kernel timing for the fused_add_rms_norm Mojo kernels.
# v1: runtime hidden size, global re-read in pass 2 (mirrors CUDA structure).
# v2: comptime hidden size, shared-memory row cache (2R+2W global traffic).
# ===----------------------------------------------------------------------=== #

from operations.fused_add_rms_norm import (
    _fused_add_rms_norm_kernel,
    _fused_add_rms_norm_smem_kernel,
)

from max.gpu.host import DeviceContext
from std.time import global_perf_counter_ns


def main() raises:
    comptime dtype = DType.bfloat16
    comptime width = 8
    comptime eps = 1e-6
    comptime iters = 200
    comptime H = 3584
    comptime BLOCK = 256

    var ctx = DeviceContext()
    comptime kernel_v1 = _fused_add_rms_norm_kernel[dtype, width, eps]
    comptime kernel_v2 = _fused_add_rms_norm_smem_kernel[
        dtype, width, eps, H, BLOCK
    ]

    for tokens in [512, 2048, 4096]:
        var count = tokens * H
        var x = ctx.enqueue_create_buffer[dtype](count)
        var res = ctx.enqueue_create_buffer[dtype](count)
        var w = ctx.enqueue_create_buffer[dtype](H)
        var out = ctx.enqueue_create_buffer[dtype](count)
        var res_out = ctx.enqueue_create_buffer[dtype](count)

        @inline(.always)
        def launch_v1(
            ctx: DeviceContext,
        ) raises {mut out, mut res_out, mut x, mut res, mut w, imm tokens}:
            ctx.enqueue_function[kernel_v1](
                out.unsafe_ptr(),
                res_out.unsafe_ptr(),
                x.unsafe_ptr(),
                res.unsafe_ptr(),
                w.unsafe_ptr(),
                Int64(H),
                grid_dim=(tokens,),
                block_dim=(BLOCK,),
            )

        @inline(.always)
        def launch_v2(
            ctx: DeviceContext,
        ) raises {mut out, mut res_out, mut x, mut res, mut w, imm tokens}:
            ctx.enqueue_function[kernel_v2](
                out.unsafe_ptr(),
                res_out.unsafe_ptr(),
                x.unsafe_ptr(),
                res.unsafe_ptr(),
                w.unsafe_ptr(),
                grid_dim=(tokens,),
                block_dim=(BLOCK,),
            )

        @inline(.always)
        def launch_v1_inplace(
            ctx: DeviceContext,
        ) raises {mut out, mut res_out, mut x, mut res, mut w, imm tokens}:
            # out aliases x, res_out aliases residual (vLLM in-place semantics)
            var xp = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
                x.unsafe_ptr()
            )
            var rp = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
                res.unsafe_ptr()
            )
            ctx.enqueue_function[kernel_v1](
                xp,
                rp,
                xp,
                rp,
                w.unsafe_ptr(),
                Int64(H),
                grid_dim=(tokens,),
                block_dim=(BLOCK,),
            )

        @inline(.always)
        def launch_v2_inplace(
            ctx: DeviceContext,
        ) raises {mut out, mut res_out, mut x, mut res, mut w, imm tokens}:
            var xp = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
                x.unsafe_ptr()
            )
            var rp = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
                res.unsafe_ptr()
            )
            ctx.enqueue_function[kernel_v2](
                xp,
                rp,
                xp,
                rp,
                w.unsafe_ptr(),
                grid_dim=(tokens,),
                block_dim=(BLOCK,),
            )

        for _ in range(20):
            launch_v1(ctx)
        ctx.synchronize()
        var t0 = global_perf_counter_ns()
        for _ in range(iters):
            launch_v1(ctx)
        ctx.synchronize()
        var dt1 = Float64(global_perf_counter_ns() - t0) / 1e9 / iters

        for _ in range(20):
            launch_v2(ctx)
        ctx.synchronize()
        var t1 = global_perf_counter_ns()
        for _ in range(iters):
            launch_v2(ctx)
        ctx.synchronize()
        var dt2 = Float64(global_perf_counter_ns() - t1) / 1e9 / iters

        for _ in range(20):
            launch_v1_inplace(ctx)
        ctx.synchronize()
        var t2 = global_perf_counter_ns()
        for _ in range(iters):
            launch_v1_inplace(ctx)
        ctx.synchronize()
        var dt3 = Float64(global_perf_counter_ns() - t2) / 1e9 / iters

        for _ in range(20):
            launch_v2_inplace(ctx)
        ctx.synchronize()
        var t3 = global_perf_counter_ns()
        for _ in range(iters):
            launch_v2_inplace(ctx)
        ctx.synchronize()
        var dt4 = Float64(global_perf_counter_ns() - t3) / 1e9 / iters

        var gbs = Float64(tokens * H * 2 * 4) / 1e9
        print(
            "tokens=",
            tokens,
            " v1=",
            dt1 * 1e6,
            "us  v2=",
            dt2 * 1e6,
            "us  v1_inplace=",
            dt3 * 1e6,
            "us (",
            gbs / dt3,
            "GB/s)  v2_inplace=",
            dt4 * 1e6,
            "us (",
            gbs / dt4,
            "GB/s)",
        )

        _ = x
        _ = res
        _ = w
        _ = out
        _ = res_out
