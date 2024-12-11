#include "time_integration.h"
#include "runge_kutta_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void update_rk3(const Block3dInfo *block_info, Block3dData *block_data,
		  const value_type dt, const size_type stage) {

    // Perform a single time step integration of the flow field equations using the third-order
    // Runge-Kutta method. 
  
    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM - 2 + num_threads.x - 1) / num_threads.x,
		    (block_info->JM - 2 + num_threads.y - 1) / num_threads.y,
		    (block_info->KM - 2 + num_threads.z - 1) / num_threads.z
		    );
  
    rk3_kernel<<< num_blocks, num_threads >>>(dt,
					      stage,
					      block_data->Jac,
					      block_data->Ep,
					      block_data->Fp,
					      block_data->Gp,
					      block_data->diff_flux_vis,
					      block_data->Q,
					      block_data->Q_p
					      );
    
    ERROR_CHECK( cudaDeviceSynchronize() );

  }
}
