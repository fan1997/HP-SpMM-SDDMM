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
#include <iostream>
#include "spmm/mm/spmm_kernel.h"
#include "spmm/mm/spmm_common.h"
#include "spmm/utils/timer_cuda.h"
#include "spmm/utils/compare.h"

#define USE_CUSPARSE_2
#define USE_CUSPARSE_3
#define USE_CUSPARSE_DEFAULT
#define USE_CUSPARSE_COO
#define USE_GESPMM
#define USE_HPSPMM
// #define CHECK_RES

int main(int argc, char **argv) {
  std::cout << "loading sparse matrix ..." << std::endl;
  ResInfo info;
  long long int num, dim, k, real_nonezero_num;
  std::string filename = argv[1];
  std::string reorder_filename = argv[2];
  info.filename_ = filename;
  std::stringstream k_str;
  k_str << argv[3];
  k_str >> k;
  info.stdv = get_standard_CV(filename);
  CSR_Matrix h_csr_mat = get_csr_mat_from_smtx(filename, num, dim, real_nonezero_num);
  COO_Matrix h_coo_mat = get_coo_mat_from_smtx(filename, num, dim, real_nonezero_num, 32);
  COO_Matrix h_coo_mat_reorder = get_coo_mat_from_smtx_reorder(filename, reorder_filename, num, dim, real_nonezero_num, 32);
  std::cout << "row_num: " << num << '\n';
  std::cout << "col_num: " << dim << '\n';
  std::cout << "k: " << k << '\n';
  std::cout << "nonzero_num: " << real_nonezero_num << '\n';
  std::cout << "row length stdv: " <<  info.stdv << '\n';
  auto density =  float(real_nonezero_num) /  (num * dim);
  std::cout << "density: " << density << '\n';
  info.num_rows_ = num;
  info.num_cols_ = dim;
  info.dense_dim_ = k;
  info.num_nnz_ = real_nonezero_num;
  // Init data
  Matrix dense_mat_cpu =  rand({dim, k}, Device(kCPU), DataType(kFloat), -1.0, 1.0);
  Matrix d_dense_mat_cuda = dense_mat_cpu.to(kCUDA);
  CSR_Matrix d_csr_mat = csr_to_cuda(h_csr_mat);
  COO_Matrix d_coo_mat = coo_to_cuda(h_coo_mat);
  COO_Matrix d_coo_mat_reorder = coo_to_cuda(h_coo_mat_reorder);
  Matrix d_result_csr_spmm_bench = full({num, k}, Device(kCUDA), DataType(kFloat), 0.0);
  Matrix d_result_csr_spmm = full({num, k}, Device(kCUDA), DataType(kFloat), 0.0);
  Matrix d_result_coo_spmm = full({num, k}, Device(kCUDA), DataType(kFloat), 0.0);
#ifdef USE_CUSPARSE_2
  float spmm_cs_csr_time = 0.0;
  cusparseCSRSpMM(d_csr_mat, d_dense_mat_cuda, d_result_csr_spmm_bench, 2, spmm_cs_csr_time);
  info.spmm_cs_csralg2_time_ = spmm_cs_csr_time;
  cudaDeviceSynchronize();
#endif

#ifdef USE_CUSPARSE_3
  spmm_cs_csr_time = 0.0;
  cusparseCSRSpMM(d_csr_mat, d_dense_mat_cuda, d_result_csr_spmm_bench, 3, spmm_cs_csr_time);
  info.spmm_cs_csralg3_time_ = spmm_cs_csr_time;
  cudaDeviceSynchronize();
#endif

#ifdef USE_CUSPARSE_DEFAULT
  spmm_cs_csr_time = 0.0;
  cusparseCSRSpMM(d_csr_mat, d_dense_mat_cuda, d_result_csr_spmm_bench, -1, spmm_cs_csr_time);
  info.spmm_cs_csrdef_time_ = spmm_cs_csr_time;
  cudaDeviceSynchronize();
#endif

#ifdef USE_CUSPARSE_COO
  float spmm_cs_coo_time = 0.0;
  cusparseCOOSpMM(d_coo_mat, d_dense_mat_cuda, d_result_coo_spmm, spmm_cs_coo_time);
  info.spmm_cs_coo_time_ = spmm_cs_coo_time;
  cudaDeviceSynchronize();
#endif

#ifdef USE_GESPMM
  float spmm_ge_time = 0.0;
  fill_matrix(d_result_csr_spmm, 0.0);
  GESpMM(d_csr_mat, d_dense_mat_cuda, d_result_csr_spmm, spmm_ge_time);
  #ifdef CHECK_RES
    ExpectEqualDenseWithError(d_result_csr_spmm, d_result_csr_spmm_bench);
  #endif
  info.spmm_ge_time_ = spmm_ge_time;
#endif

#ifdef USE_HPSPMM
  float spmm_hp_time = 0.0;
  fill_matrix(d_result_coo_spmm, 0.0);
  LBSpMM(d_coo_mat_reorder, d_dense_mat_cuda, d_result_coo_spmm, spmm_hp_time);
  info.spmm_hp_time_ = spmm_hp_time;
  #ifdef CHECK_RES
    ExpectEqualDenseWithError(d_result_coo_spmm, d_result_csr_spmm_bench, "div");
  #endif
#endif
  write_info_to_file("./full_graph_res_reorder.csv", info);
}