"""Runs vLLM's official IR-op benchmark with the Mojo provider registered,
after first verifying Mojo results against native and vllm_c.

Prerequisites:
    - A vLLM dev checkout with the `max` nightly package installed
      (``uv pip install max --extra-index-url https://whl.modular.com/nightly/simple/``)
    - The ``VLLM_ROOT`` env var pointing at that checkout (or edit below).

Usage (see README for env vars needed on driver-limited machines):
    CUDA_VISIBLE_DEVICES=1 python run_ir_bench_mojo.py
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", ""))
if not _VLLM_ROOT.is_dir():
    sys.exit("Set VLLM_ROOT to a vLLM checkout (with `max` installed in its venv)")

sys.path.insert(0, str(_VLLM_ROOT))
sys.path.insert(0, str(_HERE))

import torch

import vllm_mojo_provider  # noqa: F401  (registers the "mojo" impl)
import vllm.kernels  # noqa: F401  (registers vllm_c / oink / etc.)

import vllm.ir.ops as irops


@torch.inference_mode()
def check_correctness():
    print("=== correctness: mojo vs native vs vllm_c ===")
    torch.set_default_device("cuda")
    op = irops.fused_add_rms_norm
    for tokens, hidden in [(7, 3584), (256, 3584), (1024, 4096)]:
        kwargs = {
            "num_tokens": tokens,
            "hidden_size": hidden,
            "dtype": torch.bfloat16,
            "epsilon": 1e-5,
            "device": "cuda",
        }
        args = op.generate_inputs(**kwargs)

        ref = op.impls["native"].impl_fn(*args)
        outs = {}
        for provider in ["vllm_c", "mojo"]:
            impl = op.impls[provider]
            if not impl.supports_args(*args):
                print(f"tokens={tokens} hidden={hidden}: {provider} n/a")
                continue
            cloned = tuple(
                a.clone() if isinstance(a, torch.Tensor) else a for a in args
            )
            outs[provider] = impl.impl_fn(*cloned)
        for provider, (out, res) in outs.items():
            d_out = (out.float() - ref[0].float()).abs().max().item()
            d_res = (res.float() - ref[1].float()).abs().max().item()
            ok = torch.allclose(
                out.float(), ref[0].float(), rtol=2e-2, atol=2e-2
            )
            print(
                f"tokens={tokens:5d} hidden={hidden}  {provider}:"
                f" max|dout|={d_out:.4f} max|dres|={d_res:.4f}"
                f"  {'OK' if ok else 'MISMATCH'}"
            )
            assert ok, f"{provider} mismatch at tokens={tokens}"


def main():
    check_correctness()

    sys.argv = [
        "bench_ir_ops.py",
        "--ops",
        "fused_add_rms_norm",
        "--no-cuda-graph",
    ]
    from benchmarks.kernels.ir.bench_ir_ops import main as bench_main

    bench_main()


if __name__ == "__main__":
    main()
