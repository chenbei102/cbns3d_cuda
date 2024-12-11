#include "get_conservative.h"


namespace block3d_cuda {

  void get_conservative(const Block3dInfo *block_info, const Block3dData *block_data,
			value_type *Q) {

    // Transfer conservative variable data from device to host 
  
    size_type array_size = block_info->NEQ * block_info->IM_G * block_info->JM_G * block_info->KM_G;
    size_t d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMemcpy(Q, block_data->Q, d_size, cudaMemcpyDeviceToHost) );

  }

}
