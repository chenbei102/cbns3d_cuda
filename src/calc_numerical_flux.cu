#include "calc_numerical_flux.h"
#include "rec_riemann_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_numerical_flux(const Block3dInfo *block_info, Block3dData *block_data,
			   const value_type *Q) {

    // Reconstruct the solution at interfaces and compute the numerical flux using
    // approximate Riemann solvers. 
  
    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM - 1 + num_threads.x - 1) / num_threads.x,
		    (block_info->JM - 2 + num_threads.y - 1) / num_threads.y,
		    (block_info->KM - 2 + num_threads.z - 1) / num_threads.z
		    );
  
    rec_riemann_xi_kernel<<< num_blocks, num_threads >>>(Q,
							 block_data->rho,
							 block_data->u,
							 block_data->v,
							 block_data->w,
							 block_data->p,
							 block_data->xi_x,
							 block_data->xi_y,
							 block_data->xi_z,
							 block_data->Jac,
							 block_data->Ep
							 );

    num_blocks.x = (block_info->IM - 2 + num_threads.x - 1) / num_threads.x;
    num_blocks.y = (block_info->JM - 1 + num_threads.x - 1) / num_threads.y;
    
    rec_riemann_eta_kernel<<< num_blocks, num_threads >>>(Q,
							  block_data->rho,
							  block_data->u,
							  block_data->v,
							  block_data->w,
							  block_data->p,
							  block_data->eta_x,
							  block_data->eta_y,
							  block_data->eta_z,
							  block_data->Jac,
							  block_data->Fp
							  );

    num_blocks.y = (block_info->JM - 2 + num_threads.x - 1) / num_threads.y;
    num_blocks.z = (block_info->KM - 1 + num_threads.x - 1) / num_threads.z;
    
    rec_riemann_zeta_kernel<<< num_blocks, num_threads >>>(Q,
							   block_data->rho,
							   block_data->u,
							   block_data->v,
							   block_data->w,
							   block_data->p,
							   block_data->zeta_x,
							   block_data->zeta_y,
							   block_data->zeta_z,
							   block_data->Jac,
							   block_data->Gp
							   );
    
    ERROR_CHECK( cudaDeviceSynchronize() );

  }

}
