#ifndef _TIME_STEP_KERNEL_H_
#define _TIME_STEP_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void time_step_kernel(const value_type* rho,
				   const value_type* u,
				   const value_type* v,
				   const value_type* w,
				   const value_type* p,
				   const value_type* mu,
				   const value_type* xi_x,
				   const value_type* xi_y,
				   const value_type* xi_z,
				   const value_type* eta_x,
				   const value_type* eta_y,
				   const value_type* eta_z,
				   const value_type* zeta_x,
				   const value_type* zeta_y,
				   const value_type* zeta_z,
				   value_type* dt
				   );
  
}

#endif /* _TIME_STEP_KERNEL_H_ */
