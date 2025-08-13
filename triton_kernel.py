import torch

import triton
import triton.language as tl
import inspect
import os
from triton.language.extra.cpu import libdevice
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

from pathlib import Path
import math


DEVICE = torch.device("cpu") # triton.runtime.driver.active.get_active_torch_device()
# extern_libs = {'libdevice': '/home/jseo.kang/triton_shared/triton/third_party/nvidia/backend/lib/libdevice.10.bc'}

@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = libdevice.rsqrt(x)
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(x_ptr + offsets, y, mask=mask)


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
output_triton = torch.zeros(size, device=DEVICE)
output_torch = torch.rsqrt(x)
output_triton = torch.empty_like(x)
n_elements = output_torch.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

@triton.jit
def triton_poi_fused_cat_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38535168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 150528)
    x1 = xindex // 150528
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 151296*x1), tmp0, None)

in_ptr = torch.randn(256, 3, 224, 224).cpu()
out_ptr = torch.empty(256, 150528, 3, device=DEVICE)
grid = lambda meta: (triton.cdiv(in_ptr.numel(), 256), )
triton_poi_fused_cat_3[grid](in_ptr, out_ptr, in_ptr.numel(), XBLOCK=256)
print(out_ptr.shape)