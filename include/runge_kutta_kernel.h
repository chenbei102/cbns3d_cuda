#ifndef _RUNGE_KUTTA_KERNEL_H_
#define _RUNGE_KUTTA_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void rk3_kernel(const value_type dt,
			     const size_type stage,
			     const value_type* Jac,
			     const value_type* Ep,
			     const value_type* Fp,
			     const value_type* Gp,
			     const value_type* diff_flux_vis,
			     value_type* Q,
			     value_type* Q_p
			     );
}

#endif /* _RUNGE_KUTTA_KERNEL_H_ */
