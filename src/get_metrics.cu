#include "get_metrics.h"


namespace block3d_cuda {

  void get_metrics(const Block3dInfo *block_info, const Block3dData *block_data,
		   value_type* xi_x, value_type* xi_y, value_type* xi_z,
		   value_type* eta_x, value_type* eta_y, value_type* eta_z,
		   value_type* zeta_x, value_type* zeta_y, value_type* zeta_z,
		   value_type* Jac) {
    
    // Transfer computed geometric metrics from the device to the host

    const size_type IM = block_info->IM;
    const size_type JM = block_info->JM;
    const size_type KM = block_info->KM;

    const size_type NG = block_info->NG;

    const size_type IM_G = IM + 2*NG;
    const size_type JM_G = JM + 2*NG;
    const size_type KM_G = KM + 2*NG;

    auto copy_array = [&](const value_type* p_s, value_type* p_d) {
      for(size_type k = 0; k < KM; k++) {
	for(size_type j = 0; j < JM; j++) {
	  for(size_type i = 0; i < IM; i++) {
	    size_type idx1 = i + IM * (j + JM * k);
	    size_type idx2 = (NG+i) + IM_G * ((NG+j) + JM_G * (NG+k));
	    p_d[idx1] = p_s[idx2];
	  }
	}
      }
    };

    size_type array_size = IM_G * JM_G * KM_G;
    size_t d_size = array_size * sizeof(value_type);
    
    value_type *tmp = new value_type[array_size];

    ERROR_CHECK( cudaMemcpy(tmp, block_data->xi_x, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, xi_x);
    ERROR_CHECK( cudaMemcpy(tmp, block_data->xi_y, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, xi_y);
    ERROR_CHECK( cudaMemcpy(tmp, block_data->xi_z, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, xi_z);
  
    ERROR_CHECK( cudaMemcpy(tmp, block_data->eta_x, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, eta_x);
    ERROR_CHECK( cudaMemcpy(tmp, block_data->eta_y, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, eta_y);
    ERROR_CHECK( cudaMemcpy(tmp, block_data->eta_z, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, eta_z);
  
    ERROR_CHECK( cudaMemcpy(tmp, block_data->zeta_x, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, zeta_x);
    ERROR_CHECK( cudaMemcpy(tmp, block_data->zeta_y, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, zeta_y);
    ERROR_CHECK( cudaMemcpy(tmp, block_data->zeta_z, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, zeta_z);

    ERROR_CHECK( cudaMemcpy(tmp, block_data->Jac, d_size, cudaMemcpyDeviceToHost) );
    copy_array(tmp, Jac);

    delete[] tmp;
  
  }

}
