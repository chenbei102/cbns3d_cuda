#ifndef _GRADIENT_KERNEL_H_
#define _GRADIENT_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void gradient_kernel(const value_type* f,
				 value_type* f_xi, value_type* f_eta, value_type* f_zeta);

  __global__ void gradient_wg_kernel(const value_type* f,
				     value_type* f_xi, value_type* f_eta, value_type* f_zeta);
}

#endif /* _GRADIENT_KERNEL_H_ */
