#ifndef _WENO_H_
#define _WENO_H_

#include "constants.h"


namespace block3d_cuda {

  __device__ void rec_weno5(const value_type* f1, const value_type* f2,
			    const value_type* f3, const value_type* f4,
			    const value_type* f5,
			    const value_type R_l[constant::NEQ][constant::NEQ],
			    const value_type L_l[constant::NEQ][constant::NEQ],
			    value_type& rho_L,
			    value_type& u_L, value_type& v_L, value_type& w_L,
			    value_type& p_L);
  
}

#endif /* _WENO_H_ */
