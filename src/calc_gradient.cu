#include "calc_gradient.h"
#include "gradient_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_gradient(const Block3dInfo *block_info, const value_type* f,
		     value_type* fx, value_type* fy, value_type* fz) {

    value_type *d_f;
    value_type *d_fx;
    value_type *d_fy;
    value_type *d_fz;
  
    size_type array_size = block_info->IM * block_info->JM * block_info->KM;
    size_t d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(d_f), d_size) );
    ERROR_CHECK( cudaMemcpy(d_f, f, d_size, cudaMemcpyHostToDevice) );

    ERROR_CHECK( cudaMalloc((void **)&(d_fx), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(d_fy), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(d_fz), d_size) );

    dim3 num_threads(block3d_cuda::num_thread_x,
		     block3d_cuda::num_thread_y,
		     block3d_cuda::num_thread_z);
    dim3 num_blocks((block_info->IM + num_threads.x - 1) / num_threads.x,
		    (block_info->JM + num_threads.y - 1) / num_threads.y,
		    (block_info->KM + num_threads.z - 1) / num_threads.z
		    );
  
    block3d_cuda::gradient_kernel<<< num_blocks, num_threads >>>(d_f, d_fx, d_fy, d_fz);

    ERROR_CHECK( cudaDeviceSynchronize() );

    ERROR_CHECK( cudaMemcpy(fx, d_fx, d_size, cudaMemcpyDeviceToHost) );
    ERROR_CHECK( cudaMemcpy(fy, d_fy, d_size, cudaMemcpyDeviceToHost) );
    ERROR_CHECK( cudaMemcpy(fz, d_fz, d_size, cudaMemcpyDeviceToHost) );
  
  }

}
