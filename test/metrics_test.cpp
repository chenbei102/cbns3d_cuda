/**
 * @brief Validates metrics computation for generalized coordinate transformations.
 * 
 * This test suite validates the correctness of the `block3d_cuda::calc_metrics` function, 
 * which computes metrics supporting generalized coordinate transformations. 
 * The computed metrics are compared against analytical solutions to ensure consistency and
 * correctness.
 * 
 * @expected
 * The computed metrics values at each grid point should match the analytical
 * results within the allowable error tolerance.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <fstream>

#include "Block3d_cuda.h"
#include "copy_block_info.h"
#include "allocate_mem.h"
#include "free_mem.h"
#include "calc_metrics.h"
#include "get_metrics.h"


// some utility functions for basic matrix operations

void matrix_inv(const value_type M[9], value_type M_inv[9]) {
  value_type M_det = M[0]*M[4]*M[8] - M[0]*M[5]*M[7] -
    M[1]*M[3]*M[8] + M[1]*M[5]*M[6] + M[2]*M[3]*M[7] - M[2]*M[4]*M[6];

  if (std::abs(M_det) < 1.0e-8) {
    throw std::runtime_error("The matrix is singular.\n");
  }

  M_inv[0] = (M[4]*M[8] - M[5]*M[7]) / M_det;
  M_inv[1] = (-M[1]*M[8] + M[2]*M[7]) / M_det;
  M_inv[2] = (M[1]*M[5] - M[2]*M[4]) / M_det;
  M_inv[3] = (-M[3]*M[8] + M[5]*M[6]) / M_det;
  M_inv[4] = (M[0]*M[8] - M[2]*M[6]) / M_det;
  M_inv[5] = (-M[0]*M[5] + M[2]*M[3]) / M_det;
  M_inv[6] = (M[3]*M[7] - M[4]*M[6]) / M_det;
  M_inv[7] = (-M[0]*M[7] + M[1]*M[6]) / M_det;
  M_inv[8] = (M[0]*M[4] - M[1]*M[3]) / M_det;
}

void matrix_mul(const value_type A[9], const value_type B[9], value_type C[9]) {

  for (size_type li = 0; li < 3; li++) {
    for (size_type lj = 0; lj < 3; lj++) {

      value_type ss = 0.0;
      for (size_type lk = 0; lk < 3; lk++) {
	ss += A[3*li + lk] * B[3*lk + lj];
      }
      C[3*li + lj] = ss;

    }
  }

}

void matrix_mul_vec(const value_type A[9], const value_type b[3], value_type c[3]) {

  for (size_type li = 0; li < 3; li++) {
    value_type ss = 0.0;
    for (size_type lj = 0; lj < 3; lj++) {
      ss += A[3*li + lj] * b[lj];
    }
    c[li] = ss;
  }

}

// -----------------------------------------------------------------------------

TEST(MetricsTest, TestRectilinear) {

  size_type num_x {10};
  size_type num_y {10};
  size_type num_z {10};

  block3d_cuda::Block3dInfo block_info;

  block_info.IM = num_x;
  block_info.JM = num_y;
  block_info.KM = num_z;

  size_type num_g = block_info.NG;

  block_info.IM_G = num_x + 2*num_g; 
  block_info.JM_G = num_y + 2*num_g;
  block_info.KM_G = num_z + 2*num_g;

  block3d_cuda::copy_block_info(&block_info); 

  size_type array_size = num_x * num_y * num_z;

  value_type *x = new value_type[array_size];
  value_type *y = new value_type[array_size];
  value_type *z = new value_type[array_size];

  value_type L_x = 5.0;
  value_type L_y = 5.0;
  value_type L_z = 5.0;
  value_type s = 1.2;

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	value_type uu = std::tanh(s*(-1 + 2.0*i/(num_x-1)));
	x[idx] = 0.5*L_x*(uu + 1);

	uu = std::tanh(s*(-1 + 2.0*j/(num_y-1)));
	y[idx] = 0.5*L_y*(uu + 1);

	uu = std::tanh(s*(-1.0 + 2.0*k/(num_z-1)));
	z[idx] = 0.5*L_z*(uu + 1);

      }
    }
  }

  block3d_cuda::Block3dData block_data;

  block3d_cuda::allocate_mem(&block_info, &block_data);
  block3d_cuda::calc_metrics(&block_info, &block_data, x, y, z);
  
  value_type *xi_x;
  value_type *xi_y;
  value_type *xi_z;
  value_type *eta_x;
  value_type *eta_y;
  value_type *eta_z;
  value_type *zeta_x;
  value_type *zeta_y;
  value_type *zeta_z;

  value_type *Jac;

  xi_x = new value_type[array_size];
  xi_y = new value_type[array_size];
  xi_z = new value_type[array_size];
  eta_x = new value_type[array_size];
  eta_y = new value_type[array_size];
  eta_z = new value_type[array_size];
  zeta_x = new value_type[array_size];
  zeta_y = new value_type[array_size];
  zeta_z = new value_type[array_size];

  Jac = new value_type[array_size];

  block3d_cuda::get_metrics(&block_info, &block_data,
			    xi_x, xi_y, xi_z, eta_x, eta_y, eta_z, zeta_x, zeta_y, zeta_z,
			    Jac);
  
  value_type mse[10] = {0.0};

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	value_type uu = std::tanh(s*(-1 + 2.0*i/(num_x-1)));
	value_type x_xi = L_x*s*(1 - uu*uu)/(num_x-1);

	uu = std::tanh(s*(-1 + 2.0*j/(num_y-1)));
	value_type y_eta = L_y*s*(1 - uu*uu)/(num_y-1);

	uu = std::tanh(s*(-1.0 + 2.0*k/(num_z-1)));
	value_type z_zeta = L_z*s*(1 - uu*uu)/(num_z-1);
	
	Jac[idx] -= x_xi * y_eta * z_zeta;
	
	xi_x[idx] -= 1.0/x_xi;
	eta_y[idx] -= 1.0/y_eta;
	zeta_z[idx] -= 1.0/z_zeta;

	mse[0] += Jac[idx] * Jac[idx];

	mse[1] += xi_x[idx] * xi_x[idx];
	mse[2] += xi_y[idx] * xi_y[idx];
	mse[3] += xi_z[idx] * xi_z[idx];

	mse[4] += eta_x[idx] * eta_x[idx];
	mse[5] += eta_y[idx] * eta_y[idx];
	mse[6] += eta_z[idx] * eta_z[idx];

	mse[7] += zeta_x[idx] * zeta_x[idx];
	mse[8] += zeta_y[idx] * zeta_y[idx];
	mse[9] += zeta_z[idx] * zeta_z[idx];

      }
    }
  }

  for (size_type i = 0; i < 10; i++) {
    mse[i] /= array_size;

    EXPECT_NEAR(0.0, mse[i], 2.0e-2) << "i = " << i;
  }
  
  block3d_cuda::free_mem(&block_data);

  delete[] x;
  delete[] y;
  delete[] z;

  delete[] xi_x;
  delete[] xi_y;
  delete[] xi_z;
  delete[] eta_x;
  delete[] eta_y;
  delete[] eta_z;
  delete[] zeta_x;
  delete[] zeta_y;
  delete[] zeta_z;

  delete[] Jac;

}

TEST(MetricsTest, TestRotation) {

  size_type num_x {10};
  size_type num_y {10};
  size_type num_z {10};

  block3d_cuda::Block3dInfo block_info;

  block_info.IM = num_x;
  block_info.JM = num_y;
  block_info.KM = num_z;

  size_type num_g = block_info.NG;

  block_info.IM_G = num_x + 2*num_g; 
  block_info.JM_G = num_y + 2*num_g;
  block_info.KM_G = num_z + 2*num_g;

  block3d_cuda::copy_block_info(&block_info); 

  size_type array_size = num_x * num_y * num_z;

  value_type *x = new value_type[array_size];
  value_type *y = new value_type[array_size];
  value_type *z = new value_type[array_size];

  const value_type pi = std::acos(-1.0);
  
  value_type alpha = pi/4.0;
  value_type beta = pi/4.0;
  value_type gamma = pi/4.0;

  value_type R_mat[9] = {0.0};

  R_mat[0] = std::cos(beta)*std::cos(gamma);
  R_mat[1] = std::sin(alpha)*std::sin(beta)*std::cos(gamma) -
    std::sin(gamma)*std::cos(alpha);
  R_mat[2] = std::sin(alpha)*std::sin(gamma) +
    std::sin(beta)*std::cos(alpha)*std::cos(gamma);
  R_mat[3] = std::sin(gamma)*std::cos(beta);
  R_mat[4] = std::sin(alpha)*std::sin(beta)*std::sin(gamma) +
    std::cos(alpha)*std::cos(gamma);
  R_mat[5] = -std::sin(alpha)*std::cos(gamma) +
    std::sin(beta)*std::sin(gamma)*std::cos(alpha);
  R_mat[6] = -std::sin(beta);
  R_mat[7] = std::sin(alpha)*std::cos(beta);
  R_mat[8] = std::cos(alpha)*std::cos(beta);

  value_type coord_vec1[3] = {0.0};
  value_type coord_vec2[3] = {0.0};

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	coord_vec1[0] = i / (num_x - 1.0);
	coord_vec1[1] = j / (num_y - 1.0);
	coord_vec1[2] = k / (num_z - 1.0);

	matrix_mul_vec(R_mat, coord_vec1, coord_vec2);

	x[idx] = coord_vec2[0];
	y[idx] = coord_vec2[1];
	z[idx] = coord_vec2[2];

      }
    }
  }

  block3d_cuda::Block3dData block_data;

  block3d_cuda::allocate_mem(&block_info, &block_data);
  block3d_cuda::calc_metrics(&block_info, &block_data, x, y, z);

  value_type *xi_x;
  value_type *xi_y;
  value_type *xi_z;
  value_type *eta_x;
  value_type *eta_y;
  value_type *eta_z;
  value_type *zeta_x;
  value_type *zeta_y;
  value_type *zeta_z;

  value_type *Jac;

  xi_x = new value_type[array_size];
  xi_y = new value_type[array_size];
  xi_z = new value_type[array_size];
  eta_x = new value_type[array_size];
  eta_y = new value_type[array_size];
  eta_z = new value_type[array_size];
  zeta_x = new value_type[array_size];
  zeta_y = new value_type[array_size];
  zeta_z = new value_type[array_size];

  Jac = new value_type[array_size];

  block3d_cuda::get_metrics(&block_info, &block_data,
			    xi_x, xi_y, xi_z, eta_x, eta_y, eta_z, zeta_x, zeta_y, zeta_z,
			    Jac);
  
  value_type detJA = 0.0;
  value_type J_A[9] = {0.0};
  value_type J_B[9] = {0.0};
  value_type J_C[9] = {
    1.0/(num_x-1), 0.0, 0.0,
    0.0, 1.0/(num_y-1), 0.0,
    0.0, 0.0, 1.0/(num_z-1)
  };

  value_type mse[10] = {0.0};

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	detJA = R_mat[0]*R_mat[4]*R_mat[8] - R_mat[0]*R_mat[5]*R_mat[7] -
	  R_mat[1]*R_mat[3]*R_mat[8] + R_mat[1]*R_mat[5]*R_mat[6] +
	  R_mat[2]*R_mat[3]*R_mat[7] - R_mat[2]*R_mat[4]*R_mat[6];

	Jac[idx] -= detJA / (num_x-1) / (num_y-1) / (num_z-1);
	
	matrix_mul(R_mat, J_C, J_A);
	matrix_inv(J_A, J_B);
	
	xi_x[idx] -= J_B[0];
	xi_y[idx] -= J_B[1];
	xi_z[idx] -= J_B[2];
	eta_x[idx] -= J_B[3];
	eta_y[idx] -= J_B[4];
	eta_z[idx] -= J_B[5];
	zeta_x[idx] -= J_B[6];
	zeta_y[idx] -= J_B[7];
	zeta_z[idx] -= J_B[8];

	mse[0] += Jac[idx] * Jac[idx];

	mse[1] += xi_x[idx] * xi_x[idx];
	mse[2] += xi_y[idx] * xi_y[idx];
	mse[3] += xi_z[idx] * xi_z[idx];

	mse[4] += eta_x[idx] * eta_x[idx];
	mse[5] += eta_y[idx] * eta_y[idx];
	mse[6] += eta_z[idx] * eta_z[idx];

	mse[7] += zeta_x[idx] * zeta_x[idx];
	mse[8] += zeta_y[idx] * zeta_y[idx];
	mse[9] += zeta_z[idx] * zeta_z[idx];

      }
    }
  }

  for (size_type i = 0; i < 10; i++) {
    mse[i] /= array_size;

    EXPECT_NEAR(0.0, mse[i], 1.0e-8) << "i = " << i;
  }

  block3d_cuda::free_mem(&block_data);

  delete[] x;
  delete[] y;
  delete[] z;

  delete[] xi_x;
  delete[] xi_y;
  delete[] xi_z;
  delete[] eta_x;
  delete[] eta_y;
  delete[] eta_z;
  delete[] zeta_x;
  delete[] zeta_y;
  delete[] zeta_z;

  delete[] Jac;

}

TEST(MetricsTest, TestConicalShell) {

  size_type num_x {10};
  size_type num_y {20};
  size_type num_z {5};

  block3d_cuda::Block3dInfo block_info;

  block_info.IM = num_x;
  block_info.JM = num_y;
  block_info.KM = num_z;

  size_type num_g = block_info.NG;

  block_info.IM_G = num_x + 2*num_g; 
  block_info.JM_G = num_y + 2*num_g;
  block_info.KM_G = num_z + 2*num_g;

  block3d_cuda::copy_block_info(&block_info); 

  size_type array_size = num_x * num_y * num_z;

  value_type *x = new value_type[array_size];
  value_type *y = new value_type[array_size];
  value_type *z = new value_type[array_size];

  const value_type pi = std::acos(-1.0);
  
  value_type Rx = 3.0;
  value_type Ry = 6.0;
  value_type Rz = 5.0;
  value_type theta = 5.0*pi/12.0;
  value_type phi = 1.0*pi/12.0;

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	x[idx] = (Rx - (Rx - 1.0) * i / (num_x-1)) *
	  std::cos(theta * (2.0 * j / (num_y-1) - 1.0)) *
	  std::sin(phi * (2.0 * k / (num_z-1) + 1.0));
	y[idx] = (Ry - (Ry - 1.0) * i / (num_x-1)) *
	  std::sin(theta * (2.0 * j / (num_y-1) - 1.0)) *
	  std::sin(phi * (2.0 * k / (num_z-1) + 1.0));
	z[idx] = (Rz - (Rz - 1.0) * i / (num_x-1)) *
	  std::cos(phi * (2.0 * k / (num_z-1) + 1.0));

      }
    }
  }

  block3d_cuda::Block3dData block_data;

  block3d_cuda::allocate_mem(&block_info, &block_data);
  block3d_cuda::calc_metrics(&block_info, &block_data, x, y, z);
  
  value_type *xi_x;
  value_type *xi_y;
  value_type *xi_z;
  value_type *eta_x;
  value_type *eta_y;
  value_type *eta_z;
  value_type *zeta_x;
  value_type *zeta_y;
  value_type *zeta_z;

  value_type *Jac;

  xi_x = new value_type[array_size];
  xi_y = new value_type[array_size];
  xi_z = new value_type[array_size];
  eta_x = new value_type[array_size];
  eta_y = new value_type[array_size];
  eta_z = new value_type[array_size];
  zeta_x = new value_type[array_size];
  zeta_y = new value_type[array_size];
  zeta_z = new value_type[array_size];

  Jac = new value_type[array_size];

  block3d_cuda::get_metrics(&block_info, &block_data,
			    xi_x, xi_y, xi_z, eta_x, eta_y, eta_z, zeta_x, zeta_y, zeta_z,
			    Jac);
  
  auto calc_metrics = [&](const value_type xi,
			  const value_type eta,
			  const value_type zeta,
			  value_type& J_det, value_type J_mat[9]) {

    J_mat[0] = (1 - Rx)*std::sin(phi*(2*zeta + 1))*std::cos(theta*(2*eta - 1));
    J_mat[1] = -2*theta*(Rx - xi*(Rx - 1))*std::sin(phi*(2*zeta + 1))*
      std::sin(theta*(2*eta - 1));
    J_mat[2] = 2*phi*(Rx - xi*(Rx - 1))*std::cos(phi*(2*zeta + 1))*
      std::cos(theta*(2*eta - 1));
    J_mat[3] = (1 - Ry)*std::sin(phi*(2*zeta + 1))*std::sin(theta*(2*eta - 1));
    J_mat[4] = 2*theta*(Ry - xi*(Ry - 1))*std::sin(phi*(2*zeta + 1))*
      std::cos(theta*(2*eta - 1));
    J_mat[5] = 2*phi*(Ry - xi*(Ry - 1))*std::sin(theta*(2*eta - 1))
      *std::cos(phi*(2*zeta + 1));
    J_mat[6] = (1 - Rz)*std::cos(phi*(2*zeta + 1));
    J_mat[7] = 0;
    J_mat[8] = -2*phi*(Rz - xi*(Rz - 1))*std::sin(phi*(2*zeta + 1));

    J_det = J_mat[0]*J_mat[4]*J_mat[8] - J_mat[0]*J_mat[5]*J_mat[7] -
      J_mat[1]*J_mat[3]*J_mat[8] + J_mat[1]*J_mat[5]*J_mat[6] +
      J_mat[2]*J_mat[3]*J_mat[7] - J_mat[2]*J_mat[4]*J_mat[6];
    
  };

  value_type detJA = 0.0;
  value_type J_A[9] = {0.0};
  value_type J_B[9] = {0.0};
  value_type J_C[9] = {
    1.0/(num_x-1), 0.0, 0.0,
    0.0, 1.0/(num_y-1), 0.0,
    0.0, 0.0, 1.0/(num_z-1)
  };

  value_type mse[10] = {0.0};

  for (size_type k = 0; k < num_z; k++) {
    for (size_type j = 0; j < num_y; j++) {
      for (size_type i = 0; i < num_x; i++) {

	size_type idx = i + num_x * (j + num_y * k);

	calc_metrics(static_cast<value_type>(i)/(num_x-1),
		     static_cast<value_type>(j)/(num_y-1),
		     static_cast<value_type>(k)/(num_z-1),
		     detJA, J_B);

	Jac[idx] -= detJA / (num_x-1) / (num_y-1) / (num_z-1);
	
	matrix_mul(J_B, J_C, J_A);
	matrix_inv(J_A, J_B);
	
	xi_x[idx] -= J_B[0];
	xi_y[idx] -= J_B[1];
	xi_z[idx] -= J_B[2];
	eta_x[idx] -= J_B[3];
	eta_y[idx] -= J_B[4];
	eta_z[idx] -= J_B[5];
	zeta_x[idx] -= J_B[6];
	zeta_y[idx] -= J_B[7];
	zeta_z[idx] -= J_B[8];

	mse[0] += Jac[idx] * Jac[idx];

	mse[1] += xi_x[idx] * xi_x[idx];
	mse[2] += xi_y[idx] * xi_y[idx];
	mse[3] += xi_z[idx] * xi_z[idx];

	mse[4] += eta_x[idx] * eta_x[idx];
	mse[5] += eta_y[idx] * eta_y[idx];
	mse[6] += eta_z[idx] * eta_z[idx];

	mse[7] += zeta_x[idx] * zeta_x[idx];
	mse[8] += zeta_y[idx] * zeta_y[idx];
	mse[9] += zeta_z[idx] * zeta_z[idx];

      }
    }
  }

  for (size_type i = 0; i < 10; i++) {
    mse[i] /= array_size;

    EXPECT_NEAR(0.0, mse[i], 1.0e-3) << "i = " << i;
  }

  block3d_cuda::free_mem(&block_data);

  delete[] x;
  delete[] y;
  delete[] z;

  delete[] xi_x;
  delete[] xi_y;
  delete[] xi_z;
  delete[] eta_x;
  delete[] eta_y;
  delete[] eta_z;
  delete[] zeta_x;
  delete[] zeta_y;
  delete[] zeta_z;

  delete[] Jac;

}
