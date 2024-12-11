#ifndef _PRIMITIVE_KERNEL_H_
#define _PRIMITIVE_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void primitive_kernel(const value_type* Q,
				   value_type* rho,
				   value_type* u,
				   value_type* v,
				   value_type* w,
				   value_type* p,
				   value_type* T
				   );
  
}

#endif /* _PRIMITIVE_KERNEL_H_ */
