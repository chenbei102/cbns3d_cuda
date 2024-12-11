#ifndef _PRIMITIVE_BC_KERNEL_H_
#define _PRIMITIVE_BC_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  
  __global__ void primitive_bc_kernel(value_type* Q,
				      value_type* rho,
				      value_type* u,
				      value_type* v,
				      value_type* w,
				      value_type* p,
				      value_type* T,
				      const value_type* xi_x,
				      const value_type* xi_y,
				      const value_type* xi_z,
				      const value_type* eta_x,
				      const value_type* eta_y,
				      const value_type* eta_z,
				      const value_type* zeta_x,
				      const value_type* zeta_y,
				      const value_type* zeta_z,
				      const value_type* Jac
				      );
  
}

#endif /* _PRIMITIVE_BC_KERNEL_H_ */
