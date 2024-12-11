#ifndef _REC_RIEMANN_KERNEL_H_
#define _REC_RIEMANN_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void rec_riemann_xi_kernel(const value_type* Q,
					const value_type* rho,
					const value_type* u,
					const value_type* v,
					const value_type* w,
					const value_type* p,
					const value_type* xi_x,
					const value_type* xi_y,
					const value_type* xi_z,
					const value_type* Jac,
					value_type* Ep
					);

  __global__ void rec_riemann_eta_kernel(const value_type* Q,
					 const value_type* rho,
					 const value_type* u,
					 const value_type* v,
					 const value_type* w,
					 const value_type* p,
					 const value_type* eta_x,
					 const value_type* eta_y,
					 const value_type* eta_z,
					 const value_type* Jac,
					 value_type* Fp
					 );
  
  __global__ void rec_riemann_zeta_kernel(const value_type* Q,
					  const value_type* rho,
					  const value_type* u,
					  const value_type* v,
					  const value_type* w,
					  const value_type* p,
					  const value_type* zeta_x,
					  const value_type* zeta_y,
					  const value_type* zeta_z,
					  const value_type* Jac,
					  value_type* Gp
					  );
}

#endif /* _REC_RIEMANN_KERNEL_H_ */
