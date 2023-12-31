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
#include "spmm/mm/spmm_common.h"
#include "spmm/utils/timer_cuda.h"
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/scan.h>
#include "omp.h"
#include <random>
using namespace starml;

void write_info_to_file(std::string filename, ResInfo& info){
  std::cout << "writing info to : " << filename << std::endl;
  double spmm_flop = double(info.num_nnz_) * double(info.dense_dim_) * 2.0;
  info.spmm_cs_csralg2_throughput_ = spmm_flop/info.spmm_cs_csralg2_time_/1000./1000.;
  info.spmm_cs_csralg3_throughput_ = spmm_flop/info.spmm_cs_csralg3_time_ /1000./1000.;
  info.spmm_cs_csrdef_throughput_ = spmm_flop/info.spmm_cs_csrdef_time_ /1000./1000.;
  info.spmm_cs_coo_throughput_ = spmm_flop/info.spmm_cs_coo_time_ /1000./1000.;
  info.spmm_ge_throughput_ = spmm_flop/info.spmm_ge_time_ /1000./1000.;
  // info.spmm_sputnik_throughput_ = spmm_flop/(info.spmm_sputnik_time_ /1000./1000. );
  info.spmm_hp_throughput_ = spmm_flop/info.spmm_hp_time_ /1000./1000.;
  std::ofstream res_file;
  res_file.open(filename, std::ios::app);
  res_file << info.filename_ <<","<< info.num_rows_ << "," << info.num_cols_ << "," << info.num_nnz_ << "," << info.dense_dim_ << "," << info.stdv
           << "," <<"cuSPARSE(CSR, ALG2): " << "," << info.spmm_cs_csralg2_time_ << "," << info.spmm_cs_csralg2_throughput_ 
           << "," <<"cuSPARSE(CSR, ALG3): " << "," << info.spmm_cs_csralg3_time_ << "," << info.spmm_cs_csralg3_throughput_ 
           << "," <<"cuSPARSE(CSR, Default): " << "," << info.spmm_cs_csrdef_time_ << "," << info.spmm_cs_csrdef_throughput_ 
           << "," <<"cuSPARSE(COO): " << "," << info.spmm_cs_coo_time_ << "," << info.spmm_cs_coo_throughput_
           << "," <<"GE-SpMM: " << "," << info.spmm_ge_time_ << "," << info.spmm_ge_throughput_
          //  << "," << info.spmm_sputnik_time_ << "," << info.spmm_sputnik_pre_time_ << "," << info.spmm_sputnik_throughput_
           << "," <<"HP-SpMM " << ","<< info.spmm_hp_time_ << "," << info.spmm_hp_throughput_ 
           <<std::endl;
  res_file.close();
}

Matrix get_dense_mat_from_mtxfile(std::string filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num, bool column_major){
  cusp::coo_matrix<int, float, cusp::host_memory> coo_mat;
  cusp::io::read_matrix_market_file(coo_mat, filename);
  float *value_ptr = (float *)thrust::raw_pointer_cast(&coo_mat.values[0]);
  int *rowind_ptr = (int *)thrust::raw_pointer_cast(&coo_mat.row_indices[0]);
  int *colind_ptr = (int *)thrust::raw_pointer_cast(&coo_mat.column_indices[0]);
  num = coo_mat.num_rows;
  dim = coo_mat.num_cols;
  real_nonezero_num = coo_mat.num_entries;
  Matrix h_dense_mat = full({num, dim}, Device(kCPU), DataType(kFloat), 0.0);
  float* h_dense_mat_ptr = h_dense_mat.mutable_data<float>();
  for(int i = 0; i < real_nonezero_num; i++){
    if(column_major == false)
       h_dense_mat_ptr[ rowind_ptr[i] * dim + colind_ptr[i]] = 1.0;
    else
       h_dense_mat_ptr[ rowind_ptr[i] + colind_ptr[i] * num] = 1.0;
  }
  return h_dense_mat;
}

CSR_Matrix get_csr_mat_from_mtxfile(std::string filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num){
  std::cout << "get_csr_mat_from_mtxfile: " << filename << std::endl;
  cusp::coo_matrix<int, float, cusp::host_memory> coo_mat;
  cusp::io::read_matrix_market_file(coo_mat, filename);
  cusp::csr_matrix<int,float,cusp::host_memory> csr_mat(coo_mat);
  std::cout << "coo num: " << coo_mat.num_rows <<std::endl;
  std::cout << "coo dim: " << coo_mat.num_cols <<std::endl;
  std::cout << "coo real_nonezero_num: " << coo_mat.num_entries <<std::endl;  
  std::cout << "csr num: " << csr_mat.num_rows <<std::endl;
  std::cout << "csr dim: " << csr_mat.num_cols <<std::endl;
  std::cout << "csr real_nonezero_num: " << csr_mat.num_entries <<std::endl;  
  float *value_ptr = (float *)thrust::raw_pointer_cast(&csr_mat.values[0]);
  int *rowoff_ptr = (int *)thrust::raw_pointer_cast(&csr_mat.row_offsets[0]);
  int *colind_ptr = (int *)thrust::raw_pointer_cast(&csr_mat.column_indices[0]);
  num = csr_mat.num_rows;
  dim = csr_mat.num_cols;
  real_nonezero_num = csr_mat.num_entries;
  Matrix row_off =  Matrix({num + 1}, Device(kCPU), DataType(kInt32));
  Matrix col_ind = Matrix({real_nonezero_num}, Device(kCPU), DataType(kInt32));
  Matrix val = Matrix({real_nonezero_num}, Device(kCPU), DataType(kFloat));
  memcpy(row_off.mutable_data<int>(), rowoff_ptr, (num + 1) * sizeof(int));
  memcpy(col_ind.mutable_data<int>(), colind_ptr, real_nonezero_num * sizeof(int));
  memcpy(val.mutable_data<float>(), value_ptr, real_nonezero_num * sizeof(float));
  std::fill(val.mutable_data<float>(), val.mutable_data<float>() + real_nonezero_num, 1.0);
  CSR_Matrix csr_mat1;
  csr_mat1.column_indices_ = col_ind;
  csr_mat1.row_offsets_ = row_off;
  csr_mat1.values_ = val;
  csr_mat1.num_rows_ = num;
  csr_mat1.num_cols_ = dim;
  csr_mat1.num_nonzeros_ = real_nonezero_num;
  return csr_mat1;
}


void print_coo( const COO_Matrix& coo, std::string mode) {
  int num_rows = coo.num_rows_;
  int num_cols = coo.num_cols_;
  int nnz  = coo.num_nonzeros_;
  const int *indptr =   coo.row_indices_.data<int>();
  const int *indices = coo.column_indices_.data<int>();
  std::string file_name =  std::to_string(num_rows) + "_" + std::to_string(num_cols) + "_" + std::to_string(nnz) + "_" + mode + "_coo.smtx";
  std::ofstream outfile;
  outfile.open(file_name, std::ios::out);
  outfile << num_rows << ", " << num_cols << "," << nnz << std::endl;
  for (int i = 0; i < nnz; ++i){
    outfile << indptr[i] << " ";  // 在result.txt中写入结果
  }
  outfile << std::endl;
  for (int i = 0; i < nnz; ++i){
    outfile << indices[i] << " ";  // 在result.txt中写入结果
  }
  outfile << std::endl;
  outfile.close();
}


CSR_Matrix get_csr_mat_from_smtx(std::string filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num){
  std::cout << "loading csr mat from smtx file... " << filename << std::endl;
  std::fstream fin(filename);
  std::string readline;
  getline(fin, readline);
  std::string str_num_rows_cols_nnz = readline; 
  int n, d, nnz;
  sscanf(str_num_rows_cols_nnz.c_str(), "%d,%d,%d", &n, &d, &nnz);
  num = n;
  dim = d;
  real_nonezero_num = nnz;
  //row_off
  getline(fin, readline);
  std::string str_row_off = readline;
  //col_ind
  getline(fin, readline);
  std::string str_col_ind = readline;
  std::vector<int> rowoff;
  std::vector<int> colind;
  std::stringstream ss(str_row_off);
  std::string temp;
  while(ss >> temp){
    rowoff.push_back(std::stoi(temp));
  };
  std::stringstream ss1(str_col_ind);
  while(ss1 >> temp){
    colind.push_back(std::stoi(temp));
  };
  vector <float> values(real_nonezero_num, float(1.0));
  float *value_ptr = (float *)thrust::raw_pointer_cast(values.data());
  int *rowoff_ptr = (int *)thrust::raw_pointer_cast(rowoff.data());
  int *colind_ptr = (int *)thrust::raw_pointer_cast(colind.data());

  Matrix row_off =  Matrix({num + 1}, Device(kCPU), DataType(kInt32));
  Matrix col_ind = Matrix({real_nonezero_num}, Device(kCPU), DataType(kInt32));
  Matrix val = Matrix({real_nonezero_num}, Device(kCPU), DataType(kFloat));
  memcpy(row_off.mutable_data<int>(), rowoff_ptr, (num + 1) * sizeof(int));
  memcpy(col_ind.mutable_data<int>(), colind_ptr, real_nonezero_num * sizeof(int));
  memcpy(val.mutable_data<float>(), value_ptr, real_nonezero_num * sizeof(float));
  
  CSR_Matrix csr_mat1;
  csr_mat1.column_indices_ = col_ind;
  csr_mat1.row_offsets_ = row_off;
  csr_mat1.values_ = val;
  csr_mat1.num_rows_ = num;
  csr_mat1.num_cols_ = dim;
  csr_mat1.num_nonzeros_ = real_nonezero_num;
  return csr_mat1;
}
int roundUpToNextMultiple(int num, int multiple) {
    if (multiple == 0)
        return num;

    int remainder = num % multiple;
    if (remainder == 0)
        return num;

    return num + multiple - remainder;
}
COO_Matrix get_coo_mat_from_smtx(std::string filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num, int padding){
  std::cout << "loading coo mat from smtx file... " << filename << std::endl;
  std::fstream fin(filename);
  std::string readline;
  getline(fin, readline); // num_rows + cols + nnz
  std::string str_num_rows_cols_nnz = readline; 
  int n, d, nnz;
  sscanf(str_num_rows_cols_nnz.c_str(), "%d,%d,%d", &n, &d, &nnz);
  num = n;
  dim = d;
  real_nonezero_num = nnz;
  getline(fin, readline);   // row_off
  std::string str_row_off = readline;
  getline(fin, readline);   // col_ind
  std::string str_col_ind = readline;
  std::vector<int> rowoff;
  std::vector<int> colind;
  std::stringstream ss(str_row_off);
  std::string temp;
  while(ss >> temp) {rowoff.push_back(std::stoi(temp));};
  std::stringstream ss1(str_col_ind);
  while(ss1 >> temp) {colind.push_back(std::stoi(temp));};
  vector <float> values(real_nonezero_num, float(1.0));
  vector <int> rowind(real_nonezero_num, int(0));
  for (size_t i = 0; i < n; i++){
    int row_start = rowoff[i];
    int row_end = rowoff[i+1];
    for(int j = row_start; j < row_end; j++){
      rowind[j] = i;
    }
  }
  float *value_ptr = (float *)thrust::raw_pointer_cast(values.data());
  int *rowind_ptr = (int *)thrust::raw_pointer_cast(rowind.data());
  int *colind_ptr = (int *)thrust::raw_pointer_cast(colind.data());
  int nonezero_num_padded = roundUpToNextMultiple(real_nonezero_num, padding);
  Matrix row_ind = full({nonezero_num_padded}, Device(kCPU), DataType(kInt32), 0);
  Matrix col_ind = full({nonezero_num_padded}, Device(kCPU), DataType(kInt32), 0);
  Matrix val = full({nonezero_num_padded}, Device(kCPU), DataType(kFloat), 0.0);
  memcpy(row_ind.mutable_data<int>(), rowind_ptr, real_nonezero_num * sizeof(int));
  memcpy(col_ind.mutable_data<int>(), colind_ptr, real_nonezero_num * sizeof(int));
  memcpy(val.mutable_data<float>(), value_ptr, real_nonezero_num * sizeof(float));
  COO_Matrix coo_mat1;
  coo_mat1.column_indices_ = col_ind;
  coo_mat1.row_indices_ = row_ind;
  coo_mat1.values_ = val;
  coo_mat1.num_rows_ = num;
  coo_mat1.num_cols_ = dim;
  coo_mat1.num_nonzeros_ = nonezero_num_padded;
  return coo_mat1;
}

COO_Matrix get_coo_mat_from_smtx_reorder(std::string filename, std::string reorder_filename, long long int &num,
                                  long long int &dim, long long int &real_nonezero_num, int padding){
  std::cout << "loading coo reorderd mat from smtx file... " << filename << std::endl;
  std::fstream fin(filename);
  std::string readline;
  getline(fin, readline); // num_rows + cols + nnz
  std::string str_num_rows_cols_nnz = readline; 
  int n, d, nnz;
  sscanf(str_num_rows_cols_nnz.c_str(), "%d,%d,%d", &n, &d, &nnz);
  num = n;
  dim = d;
  real_nonezero_num = nnz;
  getline(fin, readline);
  std::string str_row_off = readline;
  getline(fin, readline);
  std::string str_col_ind = readline;
  std::vector<int> rowoff;
  std::vector<int> colind;
  std::stringstream ss(str_row_off);
  std::string temp;
  while(ss >> temp) {rowoff.push_back(std::stoi(temp));};
  std::stringstream ss1(str_col_ind);
  while(ss1 >> temp) {colind.push_back(std::stoi(temp));};
  vector <float> values(real_nonezero_num, float(1.0));
  vector <int> rowind;
  std::fstream fin1(reorder_filename);
  std::string row_order_str;
  std::vector<int> row_order;
  getline(fin1, row_order_str); // num_rows
  std::stringstream ss2(row_order_str);
  while(ss2 >> temp) {row_order.push_back(std::stoi(temp));};
  std::vector<int> colind_reorder;
  for(int i = 0; i < n; i++){
    int current_row = row_order[i];
    int start = rowoff[current_row];
    int end =  rowoff[current_row+1];
    for(int j = start; j < end; j++){
        rowind.push_back(current_row);
        colind_reorder.push_back(colind[j]);
    }
  }
  float *value_ptr = (float *)thrust::raw_pointer_cast(values.data());
  int *rowind_ptr = (int *)thrust::raw_pointer_cast(rowind.data());
  int *colind_ptr = (int *)thrust::raw_pointer_cast(colind_reorder.data());
  int nonezero_num_padded = roundUpToNextMultiple(real_nonezero_num, padding);
  Matrix row_ind = full({nonezero_num_padded}, Device(kCPU), DataType(kInt32), 0);
  Matrix col_ind = full({nonezero_num_padded}, Device(kCPU), DataType(kInt32), 0);
  Matrix val = full({nonezero_num_padded}, Device(kCPU), DataType(kFloat), 0.0);
  memcpy(row_ind.mutable_data<int>(), rowind_ptr, real_nonezero_num * sizeof(int));
  memcpy(col_ind.mutable_data<int>(), colind_ptr, real_nonezero_num * sizeof(int));
  memcpy(val.mutable_data<float>(), value_ptr, real_nonezero_num * sizeof(float));
  COO_Matrix coo_mat1;
  coo_mat1.column_indices_ = col_ind;
  coo_mat1.row_indices_ = row_ind;
  coo_mat1.values_ = val;
  coo_mat1.num_rows_ = num;
  coo_mat1.num_cols_ = dim;
  coo_mat1.num_nonzeros_ = real_nonezero_num;
  return coo_mat1;
}



double calVarStdev(std::vector<int> vecNums) {
	double sumNum = accumulate(vecNums.begin(), vecNums.end(), 0.0);
	double mean = sumNum / vecNums.size(); //均值
	double accum = 0.0;
	for_each(vecNums.begin(), vecNums.end(), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});
	double variance = accum / vecNums.size(); //方差
	double stdev = sqrt(variance); //标准差
	return stdev;
}
double calCV(std::vector<int> vecNums) {
	double sumNum = accumulate(vecNums.begin(), vecNums.end(), 0.0);
	double mean = sumNum / vecNums.size(); //均值
	double accum = 0.0;
	for_each(vecNums.begin(), vecNums.end(), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});
	double variance = accum / vecNums.size(); //方差
	double stdev = sqrt(variance); //标准差
	return stdev/mean;
}
double get_standart_cov(std::string filename) {
  std::cout << "calculate std cov... " << filename << std::endl;
  std::fstream fin(filename);
  std::string readline;
  getline(fin, readline); // num_rows+cols+nnz
  std::string str_num_rows_cols_nnz = readline; 
  int n, d, nnz;
  sscanf(str_num_rows_cols_nnz.c_str(), "%d,%d,%d", &n, &d, &nnz);
  //row_off
  getline(fin, readline);
  std::string str_row_off = readline;
  std::vector<int> rowoff;
  std::stringstream ss(str_row_off);
  std::string temp;
  while(ss >> temp){
    rowoff.push_back(std::stoi(temp));
  };
  std::vector<int> nnz_per_row;
  for (size_t i = 0; i < n; i++){
    int row_start = rowoff[i];
    int row_end = rowoff[i+1];
    int nnz_r = row_end - row_start;
    nnz_per_row.push_back(nnz_r);
  }
  auto vstdv = calVarStdev(nnz_per_row);
  return vstdv;
}
double get_standard_CV(std::string filename) {
  std::cout << "calculate std cov... " << std::endl;
  std::fstream fin(filename);
  std::string readline;
  getline(fin, readline);
  std::string str_num_rows_cols_nnz = readline; 
  int n, d, nnz;
  sscanf(str_num_rows_cols_nnz.c_str(), "%d,%d,%d", &n, &d, &nnz);
  getline(fin, readline);
  std::string str_row_off = readline;
  std::vector<int> rowoff;
  std::stringstream ss(str_row_off);
  std::string temp;
  while(ss >> temp){
    rowoff.push_back(std::stoi(temp));
  };
  std::vector<int> nnz_per_row;
  for (size_t i = 0; i < n; i++){
    int row_start = rowoff[i];
    int row_end = rowoff[i+1];
    int nnz_r = row_end - row_start;
    nnz_per_row.push_back(nnz_r);
  }
  auto CV = calCV(nnz_per_row);
  return CV;
}


CSR_Matrix csr_to_cuda(const CSR_Matrix& mat_cpu) {
   CSR_Matrix mat_cuda;
   mat_cuda.num_rows_ = mat_cpu.num_rows_;
   mat_cuda.num_cols_ = mat_cpu.num_cols_;
   mat_cuda.num_nonzeros_ =  mat_cpu.num_nonzeros_;
   mat_cuda.values_ = mat_cpu.values_.to(kCUDA);
   mat_cuda.row_offsets_ = mat_cpu.row_offsets_.to(kCUDA);
   mat_cuda.column_indices_ = mat_cpu.column_indices_.to(kCUDA);
   return mat_cuda;
}
CSR_Matrix csr_to_host(const CSR_Matrix& mat_cuda) {
   CSR_Matrix mat_cpu;
   mat_cpu.num_rows_ = mat_cuda.num_rows_;
   mat_cpu.num_cols_ = mat_cuda.num_cols_;
   mat_cpu.num_nonzeros_ =  mat_cuda.num_nonzeros_;
   mat_cpu.values_ = mat_cuda.values_.to(kCPU);
   mat_cpu.row_offsets_ = mat_cuda.row_offsets_.to(kCPU);
   mat_cpu.column_indices_ = mat_cuda.column_indices_.to(kCPU);
   return mat_cpu;
}
COO_Matrix coo_to_cuda(const COO_Matrix& mat_cpu) {
   COO_Matrix mat_cuda;
   mat_cuda.num_rows_ = mat_cpu.num_rows_;
   mat_cuda.num_cols_ = mat_cpu.num_cols_;
   mat_cuda.num_nonzeros_ =  mat_cpu.num_nonzeros_;
   mat_cuda.values_ = mat_cpu.values_.to(kCUDA);
   mat_cuda.row_indices_ = mat_cpu.row_indices_.to(kCUDA);
   mat_cuda.column_indices_ = mat_cpu.column_indices_.to(kCUDA);
   return mat_cuda;
}
COO_Matrix coo_to_host(const COO_Matrix& mat_cuda) {
   COO_Matrix mat_cpu;
   mat_cpu.num_rows_ = mat_cuda.num_rows_;
   mat_cpu.num_cols_ = mat_cuda.num_cols_;
   mat_cpu.num_nonzeros_ =  mat_cuda.num_nonzeros_;
   mat_cpu.values_ = mat_cuda.values_.to(kCPU);
   mat_cpu.row_indices_ = mat_cuda.row_indices_.to(kCPU);
   mat_cpu.column_indices_ = mat_cuda.column_indices_.to(kCPU);
   return mat_cpu;
}
Matrix csrtodense_cpu(const CSR_Matrix& Matrix1) {
  int m = Matrix1.num_rows();
  int n = Matrix1.num_cols();
  long long int matrix_size = m * n;
  Matrix h_dense_mat = Matrix({m, n}, Device(kCPU), DataType(kFloat));
  float *matrix = h_dense_mat.mutable_data<float>();
  if(matrix == NULL){
    std::cout << "out of mem" << '\n';
    exit(0);
  }
  int nnz = Matrix1.values().size();
  const int *h_rA = Matrix1.row_offsets().data<int>();
  const int *h_cA = Matrix1.column_indices().data<int>();
  const float *h_A = Matrix1.values().data<float>();
#pragma omp parallel for
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
       matrix[i*n + j] = 0;
    }
  }
  for(int i = 0; i < m; i++){
    int row_start = h_rA[i];
    int row_end = h_rA[i + 1];
    for(int j = row_start; j < row_end; j++){
      if(h_cA[j] > n){
        std::cout << "wrong" << '\n';
        exit(0);
      }
      matrix[i * n + h_cA[j]] = h_A[j];
    }
  }
  return h_dense_mat;
}
void print_csr(CSR_Matrix csr_mat) {
  std::cout << "num_rows: " << csr_mat.num_rows()<<'\n';
  std::cout << "num_cols: " << csr_mat.num_cols()<<'\n';
  std::cout << "num_nonzeros: " << csr_mat.num_nonzeros()<<'\n';
  std::cout << "row_offsets: " << '\n';
  csr_mat.row_offsets().print();
  std::cout << "column_indices: " << '\n';
  csr_mat.column_indices().print();
  std::cout << "values: " << '\n';
  csr_mat.values().print();
}
void print_coo(COO_Matrix coo_mat) {
  std::cout << "num_rows: " << coo_mat.num_rows()<<'\n';
  std::cout << "num_cols: " << coo_mat.num_cols()<<'\n';
  std::cout << "num_nonzeros: " << coo_mat.num_nonzeros()<<'\n';
  std::cout << "row_indices: " << '\n';
  coo_mat.row_indices().print();
  std::cout << "column_indices: " << '\n';
  coo_mat.column_indices().print();
  std::cout << "values: " << '\n';
  coo_mat.values().print();
}





