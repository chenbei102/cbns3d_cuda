#ifndef _CALC_GRADIENT_H_
#define _CALC_GRADIENT_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void calc_gradient(const Block3dInfo *block_info, const value_type* f,
		     value_type* fx, value_type* fy, value_type* fz);

}

#endif /* _CALC_GRADIENT_H_ */
