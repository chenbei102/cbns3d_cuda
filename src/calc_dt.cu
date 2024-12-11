#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include "calc_dt.h"
#include "time_step_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  value_type calc_dt(const Block3dInfo *block_info, Block3dData *block_data) {
    
    // Compute time step using CFL condition to ensure numerical stability

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM + num_threads.x - 1) / num_threads.x,
		    (block_info->JM + num_threads.y - 1) / num_threads.y,
		    (block_info->KM + num_threads.z - 1) / num_threads.z
		    );
    
    time_step_kernel<<< num_blocks, num_threads >>>(block_data->rho,
						    block_data->u,
						    block_data->v,
						    block_data->w,
						    block_data->p,
						    block_data->mu,
						    block_data->xi_x,
						    block_data->xi_y,
						    block_data->xi_z,
						    block_data->eta_x,
						    block_data->eta_y,
						    block_data->eta_z,
						    block_data->zeta_x,
						    block_data->zeta_y,
						    block_data->zeta_z,
						    block_data->dt
						    );

    ERROR_CHECK( cudaDeviceSynchronize() );

    const size_type array_size = block_info->IM * block_info->JM * block_info->KM;

    thrust::device_ptr<value_type> result =
      thrust::max_element(thrust::device_pointer_cast(block_data->dt),
			  thrust::device_pointer_cast(block_data->dt) + array_size);

    const value_type dt = block_info->CFL / *result;

    if(std::isnan(dt)) {
      throw std::runtime_error("An error occurred while calculating the time step.\n");
    }

    return dt;

  }

}
