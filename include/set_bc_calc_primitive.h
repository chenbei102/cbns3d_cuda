#ifndef _SET_BC_CALC_PRIMITIVE_H_
#define _SET_BC_CALC_PRIMITIVE_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void set_bc_calc_primitive(const Block3dInfo *block_info, Block3dData *block_data,
			     value_type *Q);
  
}

#endif /* _SET_BC_CALC_PRIMITIVE_H_ */
