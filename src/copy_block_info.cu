#include "copy_block_info.h"


// -----------------------------------------------------------------------------

namespace block3d_cuda {

  __constant__ Block3dInfo blk_info;
  
  extern const size_type num_thread_x = constant::THREADS_PER_BLOCK_X;
  extern const size_type num_thread_y = constant::THREADS_PER_BLOCK_Y;
  extern const size_type num_thread_z = constant::THREADS_PER_BLOCK_Z;

}
  
// -----------------------------------------------------------------------------

void block3d_cuda::copy_block_info(const Block3dInfo *block_info) {

  // Copy frequently used parameters to GPU constant memory for optimized performance.
  
  ERROR_CHECK( cudaMemcpyToSymbol((const void*)&(block3d_cuda::blk_info), (const void*)block_info,
				  sizeof(block3d_cuda::Block3dInfo)) );

}
