#ifndef TRITON_CONVERSION_GPU_BARRIER_TO_LLVM_FENCE_GPU_BARRIER_TO_LLVM_FENCE_H
#define TRITON_CONVERSION_GPU_BARRIER_TO_LLVM_FENCE_GPU_BARRIER_TO_LLVM_FENCE_H

#include "mlir/Pass/Pass.h"
#include "nppi_morphological_operations.h/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createGPUBarrierToLLVMFencePass();

}
}

#endif

