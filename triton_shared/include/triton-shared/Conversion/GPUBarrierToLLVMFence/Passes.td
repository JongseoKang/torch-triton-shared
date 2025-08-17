#ifndef GPUBARRIER_TO_LLVMFENCE_CONVERSION_PASSES
#define GPUBARRIER_TO_LLVMFENCE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def GPUBarrierToLLVMFence : Pass<"gpu_barrier-to-llvm_fence", "mlir::ModuleOp"> {
    let summary = "Convert gpu.barrier to llvm fence operation";
    let constructor = "triton::createGPUBarrierToLLVMFencePass()";
    let options = [];
}

#endif