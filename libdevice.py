import triton
import triton.language as tl
import torch
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def erf(x):
    # Coefficients for the approximation
    _a1 = tl.constexpr(0.254829592)
    _a2 = tl.constexpr(-0.284496736)
    _a3 = tl.constexpr(1.421413741)
    _a4 = tl.constexpr(-1.453152027)
    _a5 = tl.constexpr(1.061405429)
    _p  = tl.constexpr(0.3275911)
    # 1) extract sign and abs
    sign = tl.where(x >= 0.0, 1.0, -1.0)
    ax   = tl.abs(x)

    # 2) t = 1 / (1 + p * |x|)
    t = 1.0 / (1.0 + _p * ax)

    # 3) Horner polynomial evaluation
    y = _a5
    y = y * t + _a4
    y = y * t + _a3
    y = y * t + _a2
    y = y * t + _a1
    y = y * t

    # 4) multiply by exp(-x*x)
    y = y * tl.exp(-ax * ax)
    # 5) reconstruct sign: erf = sign * (1 - y)
    result = sign * (1.0 - y)
    # store back
    return result

@triton.jit
def asin(x):
    # Clamp to [-1,1] to avoid NaNs
    x = tl.clamp(x, -1.0, 1.0)

    # Initial guess y0 = x  (good for small x)
    y = x

    # 3 iterations: enough for float32 precision
    # You can tune the iteration count vs. performance.
    for _ in range(3):
        y = y - (tl.sin(y) - x) / tl.cos(y)

    return y
    

__all__ = [
    "erf",
]

# add tl.math functions
for _name in dir(tl.math):
    # skip private names and skip the ones we already did
    if _name.startswith("_") or _name in __all__ + ['List', 'T']:
        continue
    # pull it in from triton.language.math
    globals()[_name] = getattr(tl.math, _name)
    __all__.append(_name)