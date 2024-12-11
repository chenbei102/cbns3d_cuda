#ifndef _CALC_PRIMITIVE_H_
#define _CALC_PRIMITIVE_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void calc_primitive(const Block3dInfo *block_info, Block3dData *block_data,
		      const value_type *Q);
  
}

#endif /* _CALC_PRIMITIVE_H_ */
