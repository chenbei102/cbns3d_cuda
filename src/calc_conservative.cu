#include "calc_conservative.h"
#include "conservative_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_conservative(const Block3dInfo *block_info, Block3dData *block_data,
			 value_type *Q) {

    // Compute conservative variables

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM_G + num_threads.x - 1) / num_threads.x,
		    (block_info->JM_G + num_threads.y - 1) / num_threads.y,
		    (block_info->KM_G + num_threads.z - 1) / num_threads.z
		    );
    
    conservative_kernel<<< num_blocks, num_threads >>>(block_data->rho,
						       block_data->u,
						       block_data->v,
						       block_data->w,
						       block_data->p,
						       Q
						       );

    ERROR_CHECK( cudaDeviceSynchronize() );

    size_type array_size = block_info->NEQ * block_info->IM_G * block_info->JM_G * block_info->KM_G;
    size_t d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMemcpy(block_data->Q_p, block_data->Q, d_size, cudaMemcpyDeviceToDevice) );

  }

}
