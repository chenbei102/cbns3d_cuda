#ifndef _EIGENVECTORS_ROE_H_
#define _EIGENVECTORS_ROE_H_

#include "constants.h"


namespace block3d_cuda {

  __device__ void calc_eigenvectors_roe(const value_type rho_L,
					const value_type u_L,
					const value_type v_L,
					const value_type w_L,
					const value_type p_L,
					const value_type rho_R,
					const value_type u_R,
					const value_type v_R,
					const value_type w_R,
					const value_type p_R,
					value_type n_x, value_type n_y, value_type n_z, 
					value_type R[constant::NEQ][constant::NEQ],
					value_type L[constant::NEQ][constant::NEQ]
					);
  
}

#endif /* _EIGENVECTORS_ROE_H_ */
