# mojo-vllm-ops

A Mojo port of vLLM's `fused_add_rms_norm` CUDA kernel, registered into vLLM's
IR-op dispatch system as a `"mojo"` provider (alongside `"vllm_c"` and
`"native"`). No vLLM source changes required. Benchmarked on A100-SXM4-40GB.

## Results

Kernel vs kernel, same GPU, in-place, `hidden=3584`, bf16, warm L2:

| tokens | CUDA | Mojo  |
|-------:|-----:|------:|
|    512 | 7.67 µs | 8.55 µs |
|   2048 | 31.56 µs | 32.81 µs |
|   4096 | 91.65 µs | 93.04 µs |

Parity within 1.5-11% on a first port. The CUDA kernel also degrades badly at
some hidden sizes where Mojo doesn't: at 16384 tokens, `hidden=7168` runs
4972 µs (CUDA) vs 1384 µs (Mojo), and `hidden=16384` runs 8949 µs vs 3476 µs.

## The catch

The kernel is fine; the bridge isn't. Through vLLM's eager dispatch the Mojo
op pays ~85-190 µs per call (dlpack conversion per tensor + MAX engine
dispatch + cross-stream syncs) vs ~8 µs for `torch.ops._C.*`, so a ~4 µs
kernel is irrelevant below ~1000 tokens. MAX ops also cannot join torch
CUDA-graph capture — they execute on MAX's internal stream.

Takeaway: Mojo kernels are competitive and the IR-op provider seam is clean,
but per-op injection into eager vLLM negates the wins. The dispatch cost only
amortizes when the whole model runs under MAX's graph runtime (`max serve`).

## Files

- `operations/fused_add_rms_norm.mojo` — kernel + custom op
- `bench.py` — correctness vs fp32 reference (torch path)
- `bench_kernel.mojo` — standalone kernel benchmark, no torch dispatch
- `vllm_mojo_provider.py` — vLLM IR-op provider registration
- `run_ir_bench_mojo.py` — correctness vs native/vllm_c + vLLM's IR bench

## Gaps vs the CUDA kernel

Contiguous 2D inputs only (no strided rows, no rank > 2 flattening), no
`weight=None`, no scalar fallback for unaligned or odd hidden sizes, bf16
verified only, no batch-invariant mode.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python max "torch==2.13.0+cu129" \
    --extra-index-url https://whl.modular.com/nightly/simple/ \
    --index-url https://download.pytorch.org/whl/cu129
```

On RHEL with old system libstdc++, point `LD_LIBRARY_PATH` at a newer copy.
On driver < 580 (CUDA 13), set `MODULAR_NVPTX_COMPILER_PATH` to a CUDA 12
`ptxas` (torch bundles one under `nvidia/cuda_nvcc/bin/ptxas`).

```bash
python bench.py                        # correctness
.venv/bin/mojo bench_kernel.mojo -I .  # kernel benchmark
VLLM_ROOT=/path/to/vllm python run_ir_bench_mojo.py   # vLLM integration
```

The vLLM checkout for the last command needs `max` installed in its venv.

## License

Apache-2.0
