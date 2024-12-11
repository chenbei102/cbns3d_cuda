#include "set_bc_calc_primitive.h"
#include "primitive_bc_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void set_bc_calc_primitive(const Block3dInfo *block_info, Block3dData *block_data,
			     value_type *Q) {

    // Apply boundary conditions and compute primitive variables

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM + num_threads.x - 1) / num_threads.x,
		    (block_info->JM + num_threads.y - 1) / num_threads.y,
		    (block_info->KM + num_threads.z - 1) / num_threads.z
		    );
  
    primitive_bc_kernel<<< num_blocks, num_threads >>>(Q,
						       block_data->rho,
						       block_data->u,
						       block_data->v,
						       block_data->w,
						       block_data->p,
						       block_data->T,
						       block_data->xi_x,
						       block_data->xi_y,
						       block_data->xi_z,
						       block_data->eta_x,
						       block_data->eta_y,
						       block_data->eta_z,
						       block_data->zeta_x,
						       block_data->zeta_y,
						       block_data->zeta_z,
						       block_data->Jac
						       );

    ERROR_CHECK( cudaDeviceSynchronize() );
    
  }

}
