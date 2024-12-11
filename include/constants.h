#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

// This header file defines global constants to:
//  - Avoid hardcoding values directly in the code.
//  - Facilitate future modifications and updates.

#include "data_type.h"


namespace constant {

  const size_type NEQ = 5;
  const size_type NG = 2;

  const value_type PI = 3.14159265358979323846;

  const size_type THREADS_PER_BLOCK_X = 8;
  const size_type THREADS_PER_BLOCK_Y = 8;
  const size_type THREADS_PER_BLOCK_Z = 4;

}

#endif /* _CONSTANTS_H_ */
