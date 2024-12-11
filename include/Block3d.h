#ifndef _BLOCK3D_H_
#define _BLOCK3D_H_

#include <string>

#include "Block3d_cuda.h"

/**
 * @class Block3d
 * @brief Represents a 3D computational block for solving Navier-Stokes equations numerically.
 *
 * This class provides the necessary data structures and methods for numerical solutions of
 * Navier-Stokes equations.
 */
class Block3d {

  block3d_cuda::Block3dInfo block_info;

  std::string grid_fname;
  std::string checkpoint_fname;

  // Constant to define the number of equations to solve
  static const size_type NEQ {constant::NEQ};

  // Number of ghost points (used in boundary conditions)
  static const size_type NG {constant::NG};

  // Number of points in the computational mesh 
  size_type IM; // Number of points in the i-direction
  size_type JM; // Number of points in the j-direction
  size_type KM; // Number of points in the k-direction

  size_type IM_G; 
  size_type JM_G; 
  size_type KM_G; 

  value_type CFL;

  value_type t_cur;
  value_type t_end;

  bool is_restart;

  size_type nstep_max;
  size_type checkpoint_freq;

  value_type *x;
  value_type *y;
  value_type *z;

  // Test case parameters
  value_type angle_attack;

  size_type i_begin;
  size_type i_end;

  // Physical parameters relevant to the simulation
  value_type Mach;
  value_type Re;
  value_type Mach2;
  value_type gM2;
  value_type Re_inv;

  value_type p_inf;

  value_type gamma;
  value_type Pr;
  value_type Pr_t;
  value_type gam1;
  value_type gam1_inv;
  value_type Pr_inv;
  value_type Pr_t_inv;
  value_type gPr;

  value_type C_T_inf;
  value_type C_dt_v;

  // Flow field data
  value_type *u;
  value_type *v;
  value_type *w;
  value_type *rho;
  value_type *T;
  value_type *p;

  value_type *Q;

public:
  Block3d() = default;
  Block3d(size_type num_xi, size_type num_eta, size_type num_zeta);

  void read_input();
  void read_mesh();
  void calc_metrics();

  void copy_mesh(const value_type *x_arr,
		 const value_type *y_arr,
		 const value_type *z_arr);

  void get_metrics(value_type* xi_x, value_type* xi_y, value_type* xi_z,
		   value_type* eta_x, value_type* eta_y, value_type* eta_z,
		   value_type* zeta_x, value_type* zeta_y, value_type* zeta_z,
		   value_type* Jac);
  
  void solve();

  void allocate_mem();
  void free_mem();

  void output_vtk();
  void output_bin(std::string fname);
  void read_bin(std::string fname);

  bool is_finished();

};

#endif /* _BLOCK3D_H_ */
