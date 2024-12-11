#ifndef _GET_CONSERVATIVE_H_
#define _GET_CONSERVATIVE_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void get_conservative(const Block3dInfo *block_info, const Block3dData *block_data,
			value_type *Q);
  
}

#endif /* _GET_CONSERVATIVE_H_ */
