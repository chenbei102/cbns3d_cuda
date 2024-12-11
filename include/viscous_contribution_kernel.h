#ifndef _VISCOUS_CONTRIBUTION_KERNEL_H_
#define _VISCOUS_CONTRIBUTION_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void viscous_contribution_kernel(const value_type* Ev,
					      const value_type* Fv,
					      const value_type* Gv,
					      value_type* diff_flux_vis
					      );
  
}

#endif /* _VISCOUS_CONTRIBUTION_KERNEL_H_ */
