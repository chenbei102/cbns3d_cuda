/**
 * @brief Validates gradient computation on the computational grid.
 * 
 * This test case verifies the accuracy of the gradient computed by the 
 * Block3d class against the known analytical gradients of specific test functions.
 * 
 * @expected
 * The computed gradient values at each grid point should match the analytical
 * gradient within the allowable error tolerance.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "copy_block_info.h"
#include "calc_gradient.h"


TEST(GradientTest, TestFun01) {

  size_type num_x {50};
  size_type num_y {100};
  size_type num_z {150};

  block3d_cuda::Block3dInfo block_info;

  block_info.IM = num_x;
  block_info.JM = num_y;
  block_info.KM = num_z;
  
  size_type array_size = num_x * num_y * num_z;

  value_type *f = new value_type[array_size];

  value_type *fx = new value_type[array_size];
  value_type *fy = new value_type[array_size];
  value_type *fz = new value_type[array_size];

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	f[idx] = i*i + j*j + k*k;
	
      }
    }
  }

  block3d_cuda::copy_block_info(&block_info); 
  block3d_cuda::calc_gradient(&block_info, f, fx, fy, fz);

  value_type max_diff {0.0};

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	value_type dd = std::abs(2*i - fx[idx]);
	if (dd > max_diff) max_diff = dd;

	dd = std::abs(2*j - fy[idx]);
	if (dd > max_diff) max_diff = dd;

	dd = std::abs(2*k - fz[idx]);
	if (dd > max_diff) max_diff = dd;

      }
    }
  }

  EXPECT_NEAR(0.0, max_diff, 1e-5)
    << "Test function 1 failed: maximum deviation is " << max_diff;

  delete[] f;
  
  delete[] fx;
  delete[] fy;
  delete[] fz;
}

TEST(GradientTest, TestFun02) {

  size_type num_x {50};
  size_type num_y {100};
  size_type num_z {150};

  block3d_cuda::Block3dInfo block_info;

  block_info.IM = num_x;
  block_info.JM = num_y;
  block_info.KM = num_z;
  
  size_type array_size = num_x * num_y * num_z;

  value_type *f = new value_type[array_size];

  value_type *fx = new value_type[array_size];
  value_type *fy = new value_type[array_size];
  value_type *fz = new value_type[array_size];

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	f[idx] = i * j * k;

      }
    }
  }

  block3d_cuda::copy_block_info(&block_info); 
  block3d_cuda::calc_gradient(&block_info, f, fx, fy, fz);

  value_type max_diff {0.0};

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	value_type dd = std::abs(j*k - fx[idx]);
	if (dd > max_diff) max_diff = dd;

	dd = std::abs(i*k - fy[idx]);
	if (dd > max_diff) max_diff = dd;

	dd = std::abs(i*j - fz[idx]);
	if (dd > max_diff) max_diff = dd;

      }
    }
  }

  EXPECT_NEAR(0.0, max_diff, 1e-5)
    << "Test function 2 failed: maximum deviation is " << max_diff;

  delete[] f;
  
  delete[] fx;
  delete[] fy;
  delete[] fz;
}
