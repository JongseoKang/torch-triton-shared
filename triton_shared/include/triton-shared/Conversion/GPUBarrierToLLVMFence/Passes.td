#ifndef GPUBARRIER_TO_LLVMFENCE_CONVERSION_PASSES
#define GPUBARRIER_TO_LLVMFENCE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def GPUBarrierToLLVMFence : Pass<"gpu-barrier-to-llvm-fence", "mlir::ModuleOp"> {
    let summary = "Convert gpu.barrier to llvm fence operation";
    let constructor = "triton::createGPUBarrierToLLVMFencePass()";
}

#endif