#ifndef _LF_FLUX_H_
#define _LF_FLUX_H_

#include "data_type.h"


namespace block3d_cuda {

  __device__ void lf_flux(value_type rho_R,
			  value_type u_R, value_type v_R, value_type w_R,
			  value_type p_R,
			  value_type rho_L,
			  value_type u_L, value_type v_L, value_type w_L,
			  value_type p_L,
			  value_type nx, value_type ny, value_type nz,
			  value_type *flux);
  
}

#endif /* _LF_FLUX_H_ */

