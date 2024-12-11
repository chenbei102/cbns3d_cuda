#include "get_primitive.h"


namespace block3d_cuda {

  void get_primitive(const Block3dInfo *block_info, const Block3dData *block_data,
		     value_type *rho, value_type *u, value_type *v, value_type *w, value_type *p) {

    // Transfer primitive variable data from device to host

    size_type array_size = block_info->IM_G * block_info->JM_G * block_info->KM_G;
    size_t d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMemcpy(rho, block_data->rho, d_size, cudaMemcpyDeviceToHost) );
    ERROR_CHECK( cudaMemcpy(u, block_data->u, d_size, cudaMemcpyDeviceToHost) );
    ERROR_CHECK( cudaMemcpy(v, block_data->v, d_size, cudaMemcpyDeviceToHost) );
    ERROR_CHECK( cudaMemcpy(w, block_data->w, d_size, cudaMemcpyDeviceToHost) );
    ERROR_CHECK( cudaMemcpy(p, block_data->p, d_size, cudaMemcpyDeviceToHost) );

  }

}
