#ifndef TRITON_CONVERSION_GPU_BARRIER_TO_LLVM_FENCE_GPU_BARRIER_TO_LLVM_FENCE_H
#define TRITON_CONVERSION_GPU_BARRIER_TO_LLVM_FENCE_GPU_BARRIER_TO_LLVM_FENCE_H

#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/GPUBarrierToLLVMFence/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createGPUBarrierToLLVMFencePass();

}
}

#endif

