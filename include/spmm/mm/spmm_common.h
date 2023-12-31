/***************************************************************************
 * Copyright 2023 The HP-SpMM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <string.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
//starml
#include <starml/basic/matrix.h>
#include <starml/basic/scalar.h>
#include <starml/operators/factories.h>
#include <starml/operators/binary_ops.h>
#include <starml/operators/unary_ops.h>
#include <starml/basic/handle_cuda.h>
//cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
//cusp
#include <cusp/io/matrix_market.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/ell_matrix.h>
#include <cusp/array1d.h>
//omp
#include <omp.h>
#include "spmm/utils/timer_cuda.h"
using namespace starml;
using std::cout;
using std::endl;
using std::fstream;
using std::stringstream;
using std::vector;
using std::string;



class CSR_Matrix {
public:
    int num_rows_, num_cols_, num_nonzeros_;
    Matrix row_offsets_;
    Matrix column_indices_;
    Matrix values_;
    Matrix row_offsets() const {
        return row_offsets_;
    }
    Matrix column_indices() const {
        return column_indices_;
    }
    Matrix values() const {
        return values_;
    } 
    int num_rows() const {
        return num_rows_;
    } 
    int num_cols() const {
        return num_cols_;
    }  
    int num_nonzeros() const {
        return num_nonzeros_;
    }            
};

class COO_Matrix {
public:
    int num_rows_, num_cols_, num_nonzeros_;
    Matrix row_indices_;
    Matrix column_indices_;
    Matrix values_;
    Matrix row_indices() const {
        return row_indices_;
    }
    Matrix column_indices() const {
        return column_indices_;
    }
    Matrix values() const {
        return values_;
    } 
    int num_rows() const {
        return num_rows_;
    } 
    int num_cols() const {
        return num_cols_;
    }  
    int num_nonzeros() const {
        return num_nonzeros_;
    }            
};

// For recording results
struct ResInfo{
  std::string filename_;
  float spmm_cs_csralg2_time_ = 0.; 
  float spmm_cs_csralg2_throughput_ = 0.; 
  float spmm_cs_csralg3_time_ = 0.; 
  float spmm_cs_csralg3_throughput_ = 0.; 
  float spmm_cs_csrdef_time_ = 0.; 
  float spmm_cs_csrdef_throughput_ = 0.; 
  float spmm_cs_coo_time_ = 0.;
  float spmm_cs_coo_throughput_ = 0.;
  float spmm_ge_time_ = 0.;
  float spmm_ge_throughput_ = 0.;
  float spmm_merge_time_ = 0.;
  float spmm_merge_throughput_ = 0.;
  float spmm_sputnik_time_ = 0.;
  float spmm_sputnik_pre_time_ = 0.;
  float spmm_sputnik_throughput_ = 0.;
  float spmm_hp_time_ = 0.;
  float spmm_hp_throughput_ = 0.;
  float sddmm_cs_time_ = 0.;
  float sddmm_cs_throughput_ = 0.;
  float sddmm_dgl_time_ = 0.;
  float sddmm_dgl_throughput_ = 0.;
  float sddmm_hp_time_ = 0.;
  float sddmm_hp_throughput_ = 0.;
  int num_rows_;
  int num_cols_;
  int num_nnz_;
  int dense_dim_; 
  double stdv;
};


CSR_Matrix csr_to_cuda(const CSR_Matrix& mat_cpu);
CSR_Matrix csr_to_host(const CSR_Matrix& mat_cuda);
COO_Matrix coo_to_cuda(const COO_Matrix& mat_cpu);
COO_Matrix coo_to_host(const COO_Matrix& mat_cuda);

void print_csr(CSR_Matrix csr_mat);      
void print_coo(COO_Matrix coo_mat);
double get_standart_cov(std::string filename);
double get_standard_CV(std::string filename);
Matrix get_dense_mat_from_mtxfile(std::string filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num, bool column_major = false);
CSR_Matrix get_csr_mat_from_mtxfile(std::string filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num);
CSR_Matrix get_csr_mat_from_smtx(std::string filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num);  
COO_Matrix get_coo_mat_from_smtx(std::string filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num, int padding);      
COO_Matrix get_coo_mat_from_smtx_reorder(std::string filename, std::string reorder_filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num, int padding);                     
                        
void write_info_to_file(std::string filename, ResInfo& info);   



