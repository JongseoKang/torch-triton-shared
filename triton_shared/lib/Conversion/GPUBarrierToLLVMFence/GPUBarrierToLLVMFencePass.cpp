/********************************************************/
// custom pass to use llvm::fenceOp rather gpu::barrier //
/********************************************************/

// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/TypeRange.h"
// #include "mlir/IR/Types.h"
// #include "mlir/IR/ValueRange.h"

#include "mlir/IR/BuiltinOps.h"              // For ModuleOp
#include "mlir/IR/MLIRContext.h"             // For MLIRContext
#include "mlir/Pass/Pass.h"                  // For Pass infrastructure
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // For gpu::BarrierOp
#include "mlir/Dialect/LLVMIR/LLVMDialect.h" // For LLVM::FenceOp
#include "mlir/IR/PatternMatch.h"            // For IRRewriter

#include "triton-shared/Conversion/GPUBarrierToLLVMFence/GPUBarrierToLLVMFence.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"     // for llvm debug

#define DEBUG_TYPE "gpu_barrier-to-llvm_fence"
using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/GPUBarrierToLLVMFence/Passes.h.inc"

namespace mlir{
namespace triton {
#define GEN_PASS_DEF_GPUBARRIERTOLLVMFENCE
#include "triton-shared/Conversion/GPUBarrierToLLVMFence/Passes.h.inc"
}
}

namespace {

class GPUBarrierToLLVMFencePass : public triton::impl::GPUBarrierToLLVMFenceBase<GPUBarrierToLLVMFencePass> {
    using GPUBarrierToLLVMFenceBase<GPUBarrierToLLVMFencePass>::GPUBarrierToLLVMFenceBase;

public:
    void runOnOperation() override {
        auto moduleOp = getOperation();

        SmallVector<gpu::BarrierOp> barrierOps;
        moduleOp.walk([&](gpu::BarrierOp op){
            barrierOps.push_back(op);
        });

        IRRewriter rewriter(&getContext());
        for(auto bop : barrierOps){
            rewriter.setInsertionPoint(bop);
            rewriter.replaceOpWithNewOp<LLVM::FenceOp>(
                bop,
                LLVM::AtomicOrdering::seq_cst,     // sequential consistency
                StringRef("")                       // for all thread
            );
        }

        llvm::outs() << "Converted " << barrierOps.size() 
        << " gpu.barrier operations to llvm.fence\n";

    }
};
}   // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createGPUBarrierToLLVMFencePass(){
    return std::make_unique<GPUBarrierToLLVMFencePass>();
}