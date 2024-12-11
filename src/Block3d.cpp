#include <iostream>
#include <fstream>
#include <cstdio>
#include <chrono>

#include "Block3d.h"
#include "copy_block_info.h"
#include "allocate_mem.h"
#include "free_mem.h"
#include "calc_metrics.h"
#include "initial_condition.h"
#include "calc_conservative.h"
#include "set_conservative.h"
#include "set_bc_calc_primitive.h"

#ifndef IS_INVISCID
#include "calc_primitive.h"
#endif


Block3d::Block3d(size_type num_xi, size_type num_eta, size_type num_zeta)
  : IM(num_xi), JM(num_eta), KM(num_zeta),
    IM_G(IM + 2*NG), JM_G(JM + 2*NG), KM_G(KM + 2*NG)
{}

void Block3d::read_input() {

  // Read the input file

  std::ifstream fh; 

  try {
    fh.open ("input.txt");
    
    value_type T_inf {0.0};

    fh >> grid_fname;
    std::cout << "grid_fname : " << grid_fname << std::endl;

    fh >> checkpoint_fname;
    std::cout << "checkpoint_fname : " << checkpoint_fname << std::endl;

    fh >> CFL;
    std::cout << "CFL : " << CFL << std::endl;

    fh >> t_cur;
    std::cout << "t_cur : " << t_cur << std::endl;

    fh >> t_end;
    std::cout << "t_end : " << t_end << std::endl;

    fh >> Pr;
    std::cout << "Pr : " << Pr << std::endl;

    fh >> Pr_t;
    std::cout << "Pr_t : " << Pr_t << std::endl;

    fh >> gamma;
    std::cout << "gamma : " << gamma << std::endl;

    fh >> Mach;
    std::cout << "Mach : " << Mach << std::endl;

    fh >> Re;
    std::cout << "Re : " << Re << std::endl;

    fh >> T_inf;
    std::cout << "T_inf : " << T_inf << std::endl;

    fh >> angle_attack;
    std::cout << "angle_attack : " << angle_attack << std::endl;

    fh >> nstep_max;
    std::cout << "nstep_max : " << nstep_max << std::endl;

    fh >> checkpoint_freq;
    std::cout << "checkpoint_freq : " << checkpoint_freq << std::endl;

    fh >> i_begin;
    std::cout << "i_begin : " << i_begin << std::endl;

    fh >> i_end;
    std::cout << "i_end : " << i_end << std::endl;

    fh >> is_restart;
    std::cout << "is_restart : " << is_restart << std::endl;
   
    // -------------------------------------------------------------------------
    // Precompute and store frequently used physical quantities to improve
    // computational efficiency.

    Mach2 = Mach * Mach;

    Re_inv = 1.0 / Re;
    Pr_inv = 1.0 / Pr;
    Pr_t_inv = 1.0 / Pr_t;

    gam1 = gamma - 1.0;
    gM2 = gamma * Mach2;
    gam1_inv = 1.0 / gam1;
    gPr = gamma / Pr;

    p_inf = 1.0 / gM2;

    C_T_inf = 110.5 / T_inf;
    C_dt_v = std::max(4.0 / 3.0, gamma * Pr_inv) * Re_inv;

    // -------------------------------------------------------------------------

    block_info.CFL = CFL;
    block_info.angle_attack = angle_attack;
    block_info.i_begin = i_begin;
    block_info.i_end = i_end;

    block_info.gamma = gamma;
    block_info.Mach = Mach;
    block_info.Re = Re;
    block_info.Pr = Pr;
    block_info.Pr_t = Pr_t;

    block_info.Mach2 = Mach2;
    block_info.gM2 = gM2;
    block_info.Re_inv = Re_inv;
    block_info.p_inf = p_inf;
    block_info.gam1 = gam1;
    block_info.gam1_inv = gam1_inv;
    block_info.Pr_inv = Pr_inv;
    block_info.Pr_t_inv = Pr_t_inv;
    block_info.gPr = gPr;
    block_info.C_T_inf = C_T_inf;
    block_info.C_dt_v = C_dt_v;

  }
  catch (const std::ifstream::failure& e) {
    std::cout << "Exception opening/reading file\n";
    std::cout << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  fh.close();

}

void Block3d::read_mesh() {

  // Load the mesh file (expected to be in Plot3D format for a 3D single-block grid)
  // and allocate memory for the computational domain required for flow field simulation.

  std::ifstream fh; 

  try {
    fh.open (grid_fname);
    fh >> IM;
    fh >> JM;
    fh >> KM;

    std::cout << IM << ", " << JM << ", " << KM << "\n";

    IM_G = IM + 2*NG;
    JM_G = JM + 2*NG;
    KM_G = KM + 2*NG;

    allocate_mem();
    
    size_type array_size = IM * JM * KM;

    auto read_array = [&](value_type *p) {
      for(size_type i = 0; i < array_size; i++) {
	fh >> p[i];
      }
    };

    read_array(x);
    read_array(y);
    read_array(z);

    // -------------------------------------------------------------------------

    block_info.IM = IM;
    block_info.JM = JM;
    block_info.KM = KM;

    block_info.IM_G = IM_G;
    block_info.JM_G = JM_G;
    block_info.KM_G = KM_G;
  }
  catch (const std::ifstream::failure& e) {
    std::cout << "Exception opening/reading file\n";
    std::cout << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  fh.close();

}

void Block3d::solve() {

  // Perform the flow field simulation

  read_input();
  read_mesh();

  block3d_cuda::copy_block_info(&block_info);

  block3d_cuda::Block3dData block_data;

  block3d_cuda::allocate_mem(&block_info, &block_data);

  block3d_cuda::calc_metrics(&block_info, &block_data, x, y, z);

  if (is_restart) {
    block3d_cuda::initial_condition(&block_info, &block_data);
    block3d_cuda::calc_conservative(&block_info, &block_data, block_data.Q);
  } else {
    read_bin(checkpoint_fname);
    block3d_cuda::set_conservative(&block_info, &block_data, Q);
#ifndef IS_INVISCID
    block3d_cuda::calc_primitive(&block_info, &block_data, block_data.Q);
#endif
  }
  
  block3d_cuda::set_bc_calc_primitive(&block_info, &block_data, block_data.Q);


  

  block3d_cuda::free_mem(&block_data);

  free_mem();

}

void Block3d::allocate_mem() {

  // Allocate memory on the host for the flow field simulation
  
  size_type array_size = IM * JM * KM;

  x = new value_type[array_size];
  y = new value_type[array_size];
  z = new value_type[array_size];

  array_size = IM_G * JM_G * KM_G;

  rho = new value_type[array_size];
  u = new value_type[array_size];
  v = new value_type[array_size];
  w = new value_type[array_size];
  p = new value_type[array_size];
  
  array_size = NEQ * IM_G * JM_G * KM_G;

  Q = new value_type[array_size];

}

void Block3d::free_mem() {

  // Release memory on the host

  delete[] x;
  delete[] y;
  delete[] z;

  delete[] rho;
  delete[] u;
  delete[] v;
  delete[] w;
  delete[] p;

  delete[] Q;

}

void Block3d::output_vtk() {

  // Writes flow field data to a VTK Legacy file.
  
  FILE *fh1 = std::fopen ("output.vtk", "w");

  if (fh1) {
    std::fprintf(fh1, "# vtk DataFile Version 3.0\n");
    std::fprintf(fh1, "vtk output\n");
    std::fprintf(fh1, "ASCII\n");
    std::fprintf(fh1, "DATASET STRUCTURED_GRID\n");
    std::fprintf(fh1, "DIMENSIONS %d %d %d\n", IM, JM, KM);

    size_type array_size = IM * JM * KM;

    std::fprintf(fh1, "POINTS %d double\n", array_size);

    for (int i = 0; i < array_size; i++) {
      std::fprintf(fh1, "%E %E %E\n", x[i], y[i], z[i]);
    }

    auto write_array = [&](const value_type* p) {
      for(size_type k = 0; k < KM; k++) {
	for(size_type j = 0; j < JM; j++) {
	  for(size_type i = 0; i < IM; i++) {
	    size_type idx1 = (NG+i) + IM_G * ((NG+j) + JM_G * (NG+k));
	    std::fprintf(fh1, "%E\n", p[idx1]);
	  }
	}
      }
    };

    std::fprintf(fh1, "POINT_DATA %d\n", array_size);

    std::fprintf(fh1, "FIELD FieldData 5\n");

    std::fprintf(fh1, "rho 1 %d double\n", array_size);
    write_array(rho);
    std::fprintf(fh1, "u 1 %d double\n", array_size);
    write_array(u);
    std::fprintf(fh1, "v 1 %d double\n", array_size);
    write_array(v);
    std::fprintf(fh1, "w 1 %d double\n", array_size);
    write_array(w);
    std::fprintf(fh1, "p 1 %d double\n", array_size);
    write_array(p);

    std::cout << "The output is successful!\n";

    std::fclose(fh1);
  }

}

void Block3d::output_bin(std::string fname) {

  // Serializes flow field data to a binary file.
  
  std::ofstream fh(fname, std::ios::out | std::ios::binary);

  if (!fh) {
    std::cout << "Cannot open file!" << std::endl;
    return;
  }

  fh.write((char*)&t_cur, sizeof(value_type));

  fh.write((char*)&IM_G, sizeof(size_type));
  fh.write((char*)&JM_G, sizeof(size_type));
  fh.write((char*)&KM_G, sizeof(size_type));

  size_type array_size = NEQ * IM_G * JM_G * KM_G;

  fh.write((char*)Q, array_size * sizeof(value_type));

  fh.close();
  
}

void Block3d::read_bin(std::string fname) {

  // Reads flow field data from a binary file.

  std::ifstream fh;

  try {
    fh.open(fname, std::ios::in | std::ios::binary);

    value_type d_tmp;

    fh.read((char *)&d_tmp, sizeof(value_type));
    t_cur = d_tmp;

    size_type i_tmp;
    fh.read((char *)&i_tmp, sizeof(size_type));
    if (i_tmp != IM_G) throw std::runtime_error("The dimensions do not match.\n");
    fh.read((char *)&i_tmp, sizeof(size_type));
    if (i_tmp != JM_G) throw std::runtime_error("The dimensions do not match.\n");
    fh.read((char *)&i_tmp, sizeof(size_type));
    if (i_tmp != KM_G) throw std::runtime_error("The dimensions do not match.\n");

    size_type array_size = NEQ * IM_G * JM_G * KM_G;

    fh.read((char *)Q, array_size * sizeof(value_type));

  } catch (const std::ifstream::failure& e) {
    std::cout << "Exception opening/reading file\n";
    std::cout << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  } catch (const std::runtime_error& e) {
    std::cout << e.what() << std::endl;
    fh.close();
    std::exit(EXIT_FAILURE);
  }
  
  fh.close();

}

bool Block3d::is_finished() {

  // Checks if the simulation has reached the specified final time.
  
  bool flag = false;
  if (t_cur + 1.0e-8 > t_end) {
    flag = true;
    std::cout << "Finished!\n";
  }

  return flag;
  
}
