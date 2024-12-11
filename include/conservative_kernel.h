#ifndef _CONSERVATIVE_KERNEL_H_
#define _CONSERVATIVE_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void conservative_kernel(const value_type* rho,
				      const value_type* u,
				      const value_type* v,
				      const value_type* w,
				      const value_type* p,
				      value_type* Q
				      );
  
}

#endif /* _CONSERVATIVE_KERNEL_H_ */
