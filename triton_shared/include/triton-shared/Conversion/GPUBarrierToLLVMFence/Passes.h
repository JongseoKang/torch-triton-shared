#ifndef TRITON_GPUBARRIER_TO_LLVMFENCE_CONVERSION_PASSES_H
#define TRITON_GPUBARRIER_TO_LLVMFENCE_CONVERSION_PASSES_H

#include "triton-shared/Conversion/GPUBarrierToLLVMFence/GPUBarrierToLLVMFence.h"

namespace mlir [
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/GPUBarrierToLLVMFence/Passes.h.inc"

}
]

#endif
