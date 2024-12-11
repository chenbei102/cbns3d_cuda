#ifndef _TIME_INTEGRATION_H_
#define _TIME_INTEGRATION_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void update_rk3(const Block3dInfo *block_info, Block3dData *block_data,
		  const value_type dt, const size_type stage);
  
}

#endif /* _TIME_INTEGRATION_H_ */
