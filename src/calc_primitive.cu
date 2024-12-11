#include "calc_primitive.h"
#include "primitive_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_primitive(const Block3dInfo *block_info, Block3dData *block_data,
		      const value_type *Q) {
    
    // Compute primitive variables

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM_G + num_threads.x - 1) / num_threads.x,
		    (block_info->JM_G + num_threads.y - 1) / num_threads.y,
		    (block_info->KM_G + num_threads.z - 1) / num_threads.z
		    );
    
    primitive_kernel<<< num_blocks, num_threads >>>(Q,
						    block_data->rho,
						    block_data->u,
						    block_data->v,
						    block_data->w,
						    block_data->p,
						    block_data->T
						    );

    ERROR_CHECK( cudaDeviceSynchronize() );

  }

}
