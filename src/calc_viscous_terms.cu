#include "calc_viscous_terms.h"
#include "gradient_kernel.h"
#include "viscous_terms_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_viscous_terms(const Block3dInfo *block_info, Block3dData *block_data) {

    // Compute viscosity-related terms for the Navier-Stokes equations
  
#ifndef IS_INVISCID
    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM_G + num_threads.x - 1) / num_threads.x,
		    (block_info->JM_G + num_threads.y - 1) / num_threads.y,
		    (block_info->KM_G + num_threads.z - 1) / num_threads.z
		    );
  
    gradient_wg_kernel<<< num_blocks, num_threads >>>(block_data->u,
						      block_data->u_xi,
						      block_data->u_eta,
						      block_data->u_zeta
						      );

    gradient_wg_kernel<<< num_blocks, num_threads >>>(block_data->v,
						      block_data->v_xi,
						      block_data->v_eta,
						      block_data->v_zeta
						      );
  
    gradient_wg_kernel<<< num_blocks, num_threads >>>(block_data->w,
						      block_data->w_xi,
						      block_data->w_eta,
						      block_data->w_zeta
						      );

    gradient_wg_kernel<<< num_blocks, num_threads >>>(block_data->T,
						      block_data->T_xi,
						      block_data->T_eta,
						      block_data->T_zeta
						      );
    ERROR_CHECK( cudaDeviceSynchronize() );

    viscous_terms_kernel<<< num_blocks, num_threads >>>(block_data->T,
							block_data->xi_x,
							block_data->xi_y,
							block_data->xi_z,
							block_data->eta_x,
							block_data->eta_y,
							block_data->eta_z,
							block_data->zeta_x,
							block_data->zeta_y,
							block_data->zeta_z,
							block_data->u_xi,
							block_data->u_eta,
							block_data->u_zeta,
							block_data->v_xi,
							block_data->v_eta,
							block_data->v_zeta,
							block_data->w_xi,
							block_data->w_eta,
							block_data->w_zeta,
							block_data->T_xi,
							block_data->T_eta,
							block_data->T_zeta,
							block_data->mu,
							block_data->tau_xx,
							block_data->tau_yy,
							block_data->tau_zz,
							block_data->tau_xy,
							block_data->tau_xz,
							block_data->tau_yz,
							block_data->q_x,
							block_data->q_y,
							block_data->q_z
							);
    
    ERROR_CHECK( cudaDeviceSynchronize() );
#endif
  }

}
