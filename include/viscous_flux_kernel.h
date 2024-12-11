#ifndef _VISCOUS_FLUX_KERNEL_H_
#define _VISCOUS_FLUX_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void viscous_flux_kernel(const value_type* u,
				      const value_type* v,
				      const value_type* w,
				      const value_type* Jac,
				      const value_type* xi_x,
				      const value_type* xi_y,
				      const value_type* xi_z,
				      const value_type* eta_x,
				      const value_type* eta_y,
				      const value_type* eta_z,
				      const value_type* zeta_x,
				      const value_type* zeta_y,
				      const value_type* zeta_z,
				      const value_type* tau_xx,
				      const value_type* tau_yy,
				      const value_type* tau_zz,
				      const value_type* tau_xy,
				      const value_type* tau_xz,
				      const value_type* tau_yz,
				      const value_type* q_x,
				      const value_type* q_y,
				      const value_type* q_z,
				      value_type* Ev,
				      value_type* Fv,
				      value_type* Gv
				      );
  
}

#endif /* _VISCOUS_FLUX_KERNEL_H_ */
