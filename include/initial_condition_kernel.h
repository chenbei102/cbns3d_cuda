#ifndef _INITIAL_CONDITION_KERNEL_H_
#define _INITIAL_CONDITION_KERNEL_H_

#include "data_type.h"


namespace block3d_cuda {

  __global__ void initial_condition_kernel(value_type* rho,
					   value_type* u,
					   value_type* v,
					   value_type* w,
					   value_type* p,
					   value_type* T
					   );
  
}

#endif /* _INITIAL_CONDITION_KERNEL_H_ */
