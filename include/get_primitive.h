#ifndef _GET_PRIMITIVE_H_
#define _GET_PRIMITIVE_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void get_primitive(const Block3dInfo *block_info, const Block3dData *block_data,
		     value_type *rho, value_type *u, value_type *v, value_type *w, value_type *p);
  
}

#endif /* _GET_PRIMITIVE_H_ */
