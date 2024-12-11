#include "initial_condition.h"
#include "initial_condition_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void initial_condition(const Block3dInfo *block_info, Block3dData *block_data) {

    // Initialize the flow field variables to prepare for the simulation.

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM_G + num_threads.x - 1) / num_threads.x,
		    (block_info->JM_G + num_threads.y - 1) / num_threads.y,
		    (block_info->KM_G + num_threads.z - 1) / num_threads.z
		    );
  
    initial_condition_kernel<<< num_blocks, num_threads >>>(block_data->rho,
							    block_data->u,
							    block_data->v,
							    block_data->w,
							    block_data->p,
							    block_data->T
							    );

    ERROR_CHECK( cudaDeviceSynchronize() );

  }

}
