# Mojo kernels in vLLM — an integration experiment

This repo is an end-to-end evaluation of a question: **can Mojo/MAX kernels
live inside vLLM?** It contains a Mojo port of vLLM's
`fused_add_rms_norm` CUDA kernel, exposed to PyTorch via MAX's
`CustomOpLibrary`, and registered into vLLM's IR-op dispatch system as a
first-class provider (`"mojo"`, alongside `"vllm_c"` and `"native"`) — with
zero changes to vLLM source.

Everything here was measured on 4x NVIDIA A100-SXM4-40GB (driver 575.64.05,
CUDA 12.9).

## Results

### Kernel-level: Mojo vs CUDA (same GPU, same shapes, in-place, hidden=3584, bf16)

| tokens | CUDA (`vllm_c`, graph replay) | Mojo (standalone) | delta |
|-------:|------------------------------:|------------------:|------:|
|    512 |                        7.67 µs |            8.55 µs |  +11% |
|   2048 |                       31.56 µs |           32.81 µs |   +4% |
|   4096 |                       91.65 µs |           93.04 µs | +1.5% |

A first-pass Mojo port reaches parity with the heavily-tuned CUDA kernel
(warm-L2 conditions; see caveats below).

### Where the CUDA kernel breaks, Mojo wins

Under vLLM's own IR-op benchmark (eager, L2-flushed), the CUDA kernel falls
off a cliff at some hidden sizes while the Mojo kernel scales smoothly:

| shape (bf16)        | `vllm_c`  | `mojo`   | speedup |
|---------------------|----------:|---------:|--------:|
| 16384 tokens x 7168 | 4,972 µs | 1,384 µs |   3.6x  |
| 16384 x 16384       | 8,949 µs | 3,476 µs |   2.6x  |

### The integration cost (the real finding)

Through vLLM's eager dispatch, the Mojo op pays a **~85-190 µs fixed
per-call bridge cost** (dlpack marshalling per tensor + MAX engine
dispatch + cross-stream syncs) vs ~8 µs for `torch.ops._C.*`. At small
batch sizes this overhead dominates the ~4 µs kernel entirely. MAX custom
ops also cannot participate in torch CUDA-graph capture (they execute on
MAX's internal stream).

**Conclusion:** Mojo kernels are competitive at the kernel level and slot
into vLLM's IR-op provider registry cleanly, but per-op injection through
torch dispatch negates the kernel wins. Mojo kernels pay off when the
whole graph executes under MAX's runtime (`max serve`), where dispatch is
amortized across the model and kernels fuse — which is precisely MAX's
architecture thesis.

## What's in here

| File | Purpose |
|------|---------|
| `operations/fused_add_rms_norm.mojo` | The Mojo kernel + `@extensibility.register` custom op (DPS signature; call-site aliasing gives in-place semantics) |
| `bench.py` | Torch-side correctness harness vs an fp32 reference |
| `bench_kernel.mojo` | Standalone Mojo kernel benchmark (no torch dispatch), incl. out-of-place / in-place / smem-row-cache variants |
| `vllm_mojo_provider.py` | Registers the Mojo kernel as a vLLM IR-op provider (`"mojo"`) |
| `run_ir_bench_mojo.py` | Correctness check vs native/vllm_c + runs vLLM's official `bench_ir_ops.py` with the Mojo provider |

## Known kernel gaps vs the CUDA original

The core math and in-place semantics match `csrc/libtorch_stable/layernorm_kernels.cu`,
but this port is a specialization. It lacks: strided-row support (contiguous
2D only), rank > 2 flattening, `weight=None`, the scalar fallback for
unaligned/odd hidden sizes, fp16/fp32 correctness verification (bf16 only),
and the batch-invariant mode. See the provider's `supports_args` for what is
guarded; the rest is future work.

## Setup

```bash
# Environment with MAX nightly + torch (cu129 build for driver < 580)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python max \
    "torch==2.13.0+cu129" \
    --extra-index-url https://whl.modular.com/nightly/simple/ \
    --index-url https://download.pytorch.org/whl/cu129
```

On RHEL hosts with an old system libstdc++, point `LD_LIBRARY_PATH` at a
newer copy (e.g. from nsight-systems). On machines with driver < 580
(CUDA 13), point `MODULAR_NVPTX_COMPILER_PATH` at a CUDA 12 `ptxas` (torch
wheels bundle one under `nvidia/cuda_nvcc/bin/ptxas`).

```bash
# 1. Correctness through the torch dispatch path
python bench.py

# 2. Standalone kernel benchmark (the pure Mojo-vs-CUDA numbers)
.venv/bin/mojo bench_kernel.mojo -I .

# 3. Full vLLM IR-op integration: correctness + official benchmark
VLLM_ROOT=/path/to/vllm python run_ir_bench_mojo.py
```

For (3), the vLLM venv additionally needs `max` installed
(`uv pip install max --extra-index-url https://whl.modular.com/nightly/simple/`).

## License

Apache-2.0 (the kernel structure follows Apache-2.0 licensed code from
vLLM's `layernorm_kernels.cu` and Modular's custom-op examples).
