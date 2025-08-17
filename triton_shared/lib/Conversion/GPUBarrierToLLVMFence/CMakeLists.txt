add_triton_library(GPUBarrierToLLVMFence
  GPUBarrierToLLVMFencePass.cpp
  
  DEPENDS
  GPUBarrierToLLVMFenceConversionPassIncGen
  
  LINK_LIBS PUBLIC
  # Core MLIR libraries
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  
  # # Dialect libraries (based on your includes)
  # MLIRGPU                    # For gpu::BarrierOp
  # MLIRLLVMIR                 # For LLVM::FenceOp
  
  # Triton libraries (if needed)
  TritonIR
)