#include "set_conservative.h"


namespace block3d_cuda {

  void set_conservative(const Block3dInfo *block_info, Block3dData *block_data,
			const value_type *Q) {

    // Transfer conservative variable data from host to device 
  
    size_type array_size = block_info->NEQ * block_info->IM_G * block_info->JM_G * block_info->KM_G;
    size_t d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMemcpy(block_data->Q, Q, d_size, cudaMemcpyHostToDevice) );

    ERROR_CHECK( cudaMemcpy(block_data->Q_p, block_data->Q, d_size, cudaMemcpyDeviceToDevice) );

  }

}
