
include "mlir/Pass/PassBase.td"

def GPUBarrierToLLVMFence : Pass<"GPU-barrier-to-LLVM-fence", "mlir::ModuleOp"> {
  let summary = "Convert gpu.barrier to llvm fence operation";
}
