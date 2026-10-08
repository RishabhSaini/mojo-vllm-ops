"""Correctness harness for the in-place Mojo fused_add_rms_norm op.

Mirrors vLLM semantics: op(x, residual, weight) mutates x and residual
in place. Timing is done by the standalone Mojo bench (bench_kernel.mojo);
this validates numerics through the full torch dispatch path.
"""

from pathlib import Path

import torch
from max.experimental.torch import CustomOpLibrary

ops = CustomOpLibrary(Path(__file__).parent / "operations")
fused_add_rms_norm = ops.fused_add_rms_norm_mojo[{"epsilon_micro": 1}]

H = 3584
EPS = 1e-6


def reference(x, res, w):
    """fp32 reference of vLLM's fused_add_rms_norm semantics."""
    res_f = x.float() + res.float()
    var = res_f.pow(2).mean(-1, keepdim=True)
    out = res_f * torch.rsqrt(var + EPS) * w.float()
    return out.to(x.dtype), res_f.to(x.dtype)


@torch.inference_mode()
def check_correctness():
    print("=== correctness (in-place, vs fp32 reference) ===")
    for tokens in [1, 7, 256, 1024]:
        torch.manual_seed(0)
        x = (torch.randn(tokens, H, device="cuda") * 0.5).to(torch.bfloat16)
        res = (torch.randn(tokens, H, device="cuda") * 0.5).to(torch.bfloat16)
        w = (torch.rand(H, device="cuda") + 0.5).to(torch.bfloat16)

        x_orig = x.clone()
        res_orig = res.clone()

        # Call-site aliasing: outputs are the same tensors as inputs, which
        # gives vLLM's in-place semantics (x and residual mutated in place).
        fused_add_rms_norm(x, res, x, res, w)
        torch.cuda.synchronize()

        ref_out, ref_res = reference(x_orig, res_orig, w)
        d_out = (x.float() - ref_out.float()).abs().max().item()
        d_res = (res.float() - ref_res.float()).abs().max().item()
        ok = torch.allclose(x.float(), ref_out.float(), rtol=2e-2, atol=2e-2)
        print(
            f"tokens={tokens:5d}  max|dout|={d_out:.4f}  max|dres|={d_res:.4f}"
            f"  {'OK' if ok else 'MISMATCH'}"
        )
        assert ok, f"correctness failed at tokens={tokens}"


if __name__ == "__main__":
    torch.set_default_device("cuda")
    check_correctness()
    print("all correctness checks passed")
