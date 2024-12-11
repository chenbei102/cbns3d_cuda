#include "calc_viscous_flux_contribution.h"
#include "viscous_flux_kernel.h"
#include "viscous_contribution_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_viscous_flux_contribution(const Block3dInfo *block_info, Block3dData *block_data) {

    // Calculates the contribution of the derivatives of viscous fluxes for the
    // Navier-Stokes equations
  
#ifndef IS_INVISCID
    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM_G - 2 + num_threads.x - 1) / num_threads.x,
		    (block_info->JM_G - 2 + num_threads.y - 1) / num_threads.y,
		    (block_info->KM_G - 2 + num_threads.z - 1) / num_threads.z
		    );
  
    viscous_flux_kernel<<< num_blocks, num_threads >>>(block_data->u,
						       block_data->v,
						       block_data->w,
						       block_data->Jac,
						       block_data->xi_x,
						       block_data->xi_y,
						       block_data->xi_z,
						       block_data->eta_x,
						       block_data->eta_y,
						       block_data->eta_z,
						       block_data->zeta_x,
						       block_data->zeta_y,
						       block_data->zeta_z,
						       block_data->tau_xx,
						       block_data->tau_yy,
						       block_data->tau_zz,
						       block_data->tau_xy,
						       block_data->tau_xz,
						       block_data->tau_yz,
						       block_data->q_x,
						       block_data->q_y,
						       block_data->q_z,
						       block_data->Ev,
						       block_data->Fv,
						       block_data->Gv
						       );

    ERROR_CHECK( cudaDeviceSynchronize() );

    num_blocks.x = (block_info->IM - 2 + num_threads.x - 1) / num_threads.x;
    num_blocks.y = (block_info->JM - 2 + num_threads.x - 1) / num_threads.y;
    num_blocks.z = (block_info->KM - 2 + num_threads.x - 1) / num_threads.z;

    viscous_contribution_kernel<<< num_blocks, num_threads >>>(block_data->Ev,
							       block_data->Fv,
							       block_data->Gv,
							       block_data->diff_flux_vis
							       );
    
    ERROR_CHECK( cudaDeviceSynchronize() );
#endif
  }

}
