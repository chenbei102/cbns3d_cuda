#ifndef _VISCOUS_TERMS_KERNEL_H_
#define _VISCOUS_TERMS_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void viscous_terms_kernel(const value_type* T,
				       const value_type* xi_x,
				       const value_type* xi_y,
				       const value_type* xi_z,
				       const value_type* eta_x,
				       const value_type* eta_y,
				       const value_type* eta_z,
				       const value_type* zeta_x,
				       const value_type* zeta_y,
				       const value_type* zeta_z,
				       const value_type* u_xi,
				       const value_type* u_eta,
				       const value_type* u_zeta,
				       const value_type* v_xi,
				       const value_type* v_eta,
				       const value_type* v_zeta,
				       const value_type* w_xi,
				       const value_type* w_eta,
				       const value_type* w_zeta,
				       const value_type* T_xi,
				       const value_type* T_eta,
				       const value_type* T_zeta,
				       value_type* mu,
				       value_type* tau_xx,
				       value_type* tau_yy,
				       value_type* tau_zz,
				       value_type* tau_xy,
				       value_type* tau_xz,
				       value_type* tau_yz,
				       value_type* q_x,
				       value_type* q_y,
				       value_type* q_z
				       );
  
}

#endif /* _VISCOUS_TERMS_KERNEL_H_ */
