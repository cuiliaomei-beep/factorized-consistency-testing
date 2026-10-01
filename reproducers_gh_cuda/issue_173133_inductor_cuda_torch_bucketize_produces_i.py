# source: https://github.com/pytorch/pytorch/issues/173133
# title: [Inductor] [CUDA] torch.bucketize produces inconsistent results for nan values between Eager and Inductor modes on CUDA
# state: closed  created: 2026-01-23
# mined automatically; the harness records the torch.compile target and its first call

import torch

# Run on CUDA
x = torch.tensor([-1.0], device="cuda")  # a negative number produces nan
thresholds = torch.tensor([0.2, 0.5, 0.8], device="cuda")

# Eager mode
result_eager = torch.bucketize(torch.rsqrt(x), thresholds, right=True)
# Output: tensor([3], device='cuda:0')
print(result_eager)
# Inductor mode
@torch.compile(backend="inductor")
def compiled_func(x, thr):
    return torch.bucketize(torch.rsqrt(x), thr, right=True)

result_inductor = compiled_func(x, thresholds)
print(result_inductor)
# Output: tensor([0], device='cuda:0')
# ❌ Results are inconsistent!
