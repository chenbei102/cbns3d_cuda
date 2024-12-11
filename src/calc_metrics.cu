#include "calc_metrics.h"
#include "gradient_kernel.h"
#include "metrics_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern const size_type num_thread_x;
  extern const size_type num_thread_y;
  extern const size_type num_thread_z;

  // ---------------------------------------------------------------------------

  void calc_metrics(const Block3dInfo *block_info, Block3dData *block_data,
		    const value_type *x, const value_type *y, const value_type *z) {

    // Calculate geometric parameters for coordinate transformation

    // Transfer coordinate data from host memory to device memory
  
    size_type array_size = block_info->IM * block_info->JM * block_info->KM;
    size_t d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->x), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->y), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->z), d_size) );
  
    ERROR_CHECK( cudaMemcpy(block_data->x, x, d_size, cudaMemcpyHostToDevice) );
    ERROR_CHECK( cudaMemcpy(block_data->y, y, d_size, cudaMemcpyHostToDevice) );
    ERROR_CHECK( cudaMemcpy(block_data->z, z, d_size, cudaMemcpyHostToDevice) );
  
    ERROR_CHECK( cudaMalloc((void **)&(block_data->x_xi), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->y_xi), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->z_xi), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->x_eta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->y_eta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->z_eta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->x_zeta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->y_zeta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->z_zeta), d_size) );

    dim3 num_threads(num_thread_x, num_thread_y, num_thread_z);
    dim3 num_blocks((block_info->IM + num_threads.x - 1) / num_threads.x,
		    (block_info->JM + num_threads.y - 1) / num_threads.y,
		    (block_info->KM + num_threads.z - 1) / num_threads.z
		    );
  
    gradient_kernel<<< num_blocks, num_threads >>>(block_data->x,
						   block_data->x_xi,
						   block_data->x_eta,
						   block_data->x_zeta
						   );

    gradient_kernel<<< num_blocks, num_threads >>>(block_data->y,
						   block_data->y_xi,
						   block_data->y_eta,
						   block_data->y_zeta
						   );
  
    gradient_kernel<<< num_blocks, num_threads >>>(block_data->z,
						   block_data->z_xi,
						   block_data->z_eta,
						   block_data->z_zeta
						   );

    ERROR_CHECK( cudaDeviceSynchronize() );

    // -------------------------------------------------------------------------

    metrics_kernel<<< num_blocks, num_threads >>>(block_data->x_xi,
						  block_data->x_eta,
						  block_data->x_zeta,
						  block_data->y_xi,
						  block_data->y_eta,
						  block_data->y_zeta,
						  block_data->z_xi,
						  block_data->z_eta,
						  block_data->z_zeta,
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

    // -------------------------------------------------------------------------
  
    cudaFree(block_data->x_xi);
    cudaFree(block_data->y_xi);
    cudaFree(block_data->z_xi);
    cudaFree(block_data->x_eta);
    cudaFree(block_data->y_eta);
    cudaFree(block_data->z_eta);
    cudaFree(block_data->x_zeta);
    cudaFree(block_data->y_zeta);
    cudaFree(block_data->z_zeta);

    cudaFree(block_data->x);
    cudaFree(block_data->y);
    cudaFree(block_data->z);
    
  }

}
