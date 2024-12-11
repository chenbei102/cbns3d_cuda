#ifndef _METRICS_KERNEL_H_
#define _METRICS_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void metrics_kernel(const value_type* x_xi,
				 const value_type* x_eta,
				 const value_type* x_zeta,
				 const value_type* y_xi,
				 const value_type* y_eta,
				 const value_type* y_zeta,
				 const value_type* z_xi,
				 const value_type* z_eta,
				 const value_type* z_zeta,
				 value_type* xi_x,
				 value_type* xi_y,
				 value_type* xi_z,
				 value_type* eta_x,
				 value_type* eta_y,
				 value_type* eta_z,
				 value_type* zeta_x,
				 value_type* zeta_y,
				 value_type* zeta_z,
				 value_type* Jac
				 );

}

#endif /* _METRICS_KERNEL_H_ */
