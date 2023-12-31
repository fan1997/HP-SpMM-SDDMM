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
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/scan.h>
#include <numeric>
#include "spmm/mm/spmm_kernel.h"
#include "spmm/utils/timer_cuda.h"
#include <iostream>
#include "cuda_fp16.h"
// We integrated GE-SpMM's source code (https://github.com/hgyhungry/ge-spmm.git)

template<typename T>
__global__ void
fill_cuda(T* x, T val, int size) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    x[i] = val;
  }
}

void fill_matrix(Matrix& a, float val){
  fill_cuda<<<(a.size() + 255) / 256, 256>>>(a.mutable_data<float>(), val, a.size());
}

void cusparseCSRSpMM(const CSR_Matrix &mat1, const Matrix &mat2, Matrix &result, int algid, float& time) {
  GpuTimer timer;
  const int n_rows_mat1 = mat1.num_rows();
  const int n_cols_mat1 = mat1.num_cols();
  const int n_rows_mat2 = mat2.dim(0);
  const int n_cols_mat2 = mat2.dim(1);
  const int m = n_rows_mat1;
  const int k = n_cols_mat1;
  const int n = n_cols_mat2;
  using scalar_t = float;
  scalar_t alpha = 1.0;
  scalar_t beta = 0.0;
  int nnzA = mat1.values().size();
  int *rowindA_csr = mat1.row_offsets().mutable_data<int>();
  int *colindA = mat1.column_indices().mutable_data<int>();
  scalar_t *valuesA = mat1.values().mutable_data<scalar_t>();
  scalar_t *B = mat2.mutable_data<scalar_t>();
  scalar_t *C = result.mutable_data<scalar_t>();
 // cuda handle
  cusparseHandle_t cusparse_handle = 0;
  cusparseCreate(&cusparse_handle);
#if CUDART_VERSION < 11000 
  int ldb = n;
  int ldc = m;
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  // kernel
  for (int i = 0; i < WARMUP_NUM_CUDA; i++) {
  cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
    nnzA, &alpha, descr, valuesA, rowindA_csr, colindA,
    B, ldb, &beta, C, ldc);
  }
  timer.Start();
  for (int i = 0; i < EXE_NUM_CUDA; i++) {
  cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
    nnzA, &alpha, descr, valuesA, rowindA_csr, colindA,
    B, ldb, &beta, C, ldc);
  }
  timer.Stop();
  time =  (float)timer.Elapsed()/EXE_NUM_CUDA;
  std::cout << "cuSPARSE 101 time = " << time << " ms" << std::endl;
#else
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  cusparseCreateCsr(&matA,
      m, k, nnzA,
      rowindA_csr,
      colindA,
      valuesA,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnMat(&matB,
      k, n, n,
      B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  cusparseCreateDnMat(&matC,
      m, n, n,
      C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  cusparseSpMMAlg_t alg = CUSPARSE_SPMM_CSR_ALG2;
  if(algid == -1){
    alg = CUSPARSE_SPMM_ALG_DEFAULT;
  } else if(algid == 2){
    alg = CUSPARSE_SPMM_CSR_ALG2;
  } else if (algid == 3) {
    alg = CUSPARSE_SPMM_CSR_ALG3;
  }
      for (int i = 0; i < WARMUP_NUM_CUDA; i++) {
        cusparseSpMM_bufferSize(
          cusparse_handle, transA, transB,
          &alpha, matA, matB, &beta, matC,
          CUDA_R_32F, alg,
          &workspace_size);
      }
      timer.Start();
      for (int i = 0; i < EXE_NUM_CUDA; i++) {
        cusparseSpMM_bufferSize(
          cusparse_handle, transA, transB,
          &alpha, matA, matB, &beta, matC,
          CUDA_R_32F, alg,
          &workspace_size);
      }
      timer.Stop();
      std::cout << "cusparse csr buffer time: " <<  timer.Elapsed() / EXE_NUM_CUDA << " ms " << std::endl;
      void* workspace=NULL;
      cudaMalloc(&workspace, workspace_size);
      for (int i = 0; i < WARMUP_NUM_CUDA; i++) {
        cusparseSpMM(
          cusparse_handle, transA, transB,
          &alpha, matA, matB, &beta, matC,
          CUDA_R_32F,alg,
          workspace);
      }
      timer.Start();
      for (int i = 0; i < EXE_NUM_CUDA; i++) {
        cusparseSpMM(
            cusparse_handle, transA, transB,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F, alg,
            workspace);
      }
      timer.Stop();  
      time = timer.Elapsed() / EXE_NUM_CUDA;
      std::cout << "cusparse csr exe time: " <<  time << " ms " << std::endl;
      cudaFree(workspace);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnMat(matB);
  cusparseDestroyDnMat(matC);
#endif 
}


void cusparseCOOSpMM(const COO_Matrix &mat1, const Matrix &mat2, Matrix &result, float& time) {
  GpuTimer timer;
  const int n_rows_mat1 = mat1.num_rows();
  const int n_cols_mat1 = mat1.num_cols();
  const int n_rows_mat2 = mat2.dim(0);
  const int n_cols_mat2 = mat2.dim(1);
  const int m = n_rows_mat1;
  const int k = n_cols_mat1;
  const int n = n_cols_mat2;
  using scalar_t = float;
  scalar_t alpha = 1.0;
  scalar_t beta = 0.0;
  const int nnzA = mat1.values().size();
  const int *rowindA_coo = mat1.row_indices().data<int>();
  const int *colindA = mat1.column_indices().data<int>();
  const scalar_t *valuesA = mat1.values().data<scalar_t>();
  const scalar_t *B = mat2.data<scalar_t>();
  scalar_t *C = result.mutable_data<scalar_t>();
 // cuda handle
  cusparseHandle_t cusparse_handle = 0;
  cusparseCreate(&cusparse_handle);
#if CUDART_VERSION >= 11000
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  cusparseCreateCoo(&matA,
      m, k, nnzA,
      (void *)rowindA_coo,
      (void *)colindA,
      (void *)valuesA,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnMat(&matB,
      k, n, n,
      (void *)B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  cusparseCreateDnMat(&matC,
      m, n, n,
      (void *)C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;

    for (int i = 0; i < WARMUP_NUM_CUDA; i++) {
      cusparseSpMM_bufferSize(
      cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F,CUSPARSE_SPMM_COO_ALG4,
      &workspace_size);
    }
    timer.Start();
    for (int i = 0; i < EXE_NUM_CUDA; i++) {
      cusparseSpMM_bufferSize(
      cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F,CUSPARSE_SPMM_COO_ALG4,
      &workspace_size);
    }
    timer.Stop();
    std::cout << "cusparse coo buffer time: " <<  timer.Elapsed() / EXE_NUM_CUDA << " ms " << std::endl;
    void* workspace=NULL;
    cudaMalloc(&workspace, workspace_size);
    for (int i = 0; i < WARMUP_NUM_CUDA; i++) {
      cusparseSpMM(
      cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPMM_COO_ALG4,
      workspace);
    }
    timer.Start();
    for (int i = 0; i < EXE_NUM_CUDA; i++) {
      cusparseSpMM(
      cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPMM_COO_ALG4,
      workspace);
    }
    timer.Stop();
    time = timer.Elapsed() / EXE_NUM_CUDA;
    std::cout << "cusparse coo time: " << time  << " ms " << std::endl;
  cudaFree(workspace);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnMat(matB);
  cusparseDestroyDnMat(matC);
#endif 
}
__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() {
  return 0.0;
}

__global__ void topoCacheCoarsenSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* A_value, const float* B, float* C
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset+threadIdx.x;
  int value_off = blockDim.y * blockDim.x;  


  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {

    int cid = (blockIdx.y<<6)+threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid+1];
    int ptr = lb+threadIdx.x;
    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)] * B[offset]);
          acc2 = sum_reduce(acc2, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)] * B[(offset+32)]);
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
      C[offset+32] = acc2;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)] * B[offset]);}
          if (nout>1) {
          acc2 = sum_reduce(acc2, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)] * B[(offset+32)]);}
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
      if (nout>1) {
      C[offset+32] = acc2;}
    }
  }
} 

__global__ void topoCacheSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* A_value, const float* B, float* C 
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset + threadIdx.x;
  
  int cid = (blockIdx.y<<5)+threadIdx.x;
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  int value_off = blockDim.y * blockDim.x;  

  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    int ptr = lb+threadIdx.x;
    float acc1 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[sm_offset+kk]+cid;
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)]*B[offset]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)]*B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
    }
  }
}

__global__ void topoSimpleSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C 
) {
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    float acc1 = sum_init();
    int offset;
    for (int ptr=lb; ptr<hb; ptr++) {
      offset = A_indices[ptr]*k+threadIdx.x;
      acc1 = sum_reduce(acc1, B[offset]);
    }
    C[(rid*k+threadIdx.x)] = acc1;
  }
}

void GESpMM(const CSR_Matrix &mat1, const Matrix &mat2, Matrix &result, float& time) {
    GpuTimer timer;
    const int n_rows_mat1 = mat1.num_rows();
    const int n_cols_mat1 = mat1.num_cols();
    const int n_rows_mat2 = mat2.dim(0);
    const int n_cols_mat2 = mat2.dim(1);
    const int m = n_rows_mat1;
    const int k = n_cols_mat2;
    using scalar_t = float;
    const int nnzA = mat1.values().size();
    const int *rowindA_csr = mat1.row_offsets().data<int>();
    const int *colindA = mat1.column_indices().data<int>();
    const scalar_t *valuesA = mat1.values().data<scalar_t>();
    const scalar_t *B = mat2.data<scalar_t>();
    scalar_t *C = result.mutable_data<scalar_t>();
    timer.Start();
  for (int i = 0; i < EXE_NUM_CUDA; i++) {
    if (k<32) {
        const int row_per_block = 128/k;
        const int n_block = (m+row_per_block-1)/row_per_block;
        topoSimpleSPMMKernel<<< dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(
            m, k, rowindA_csr, colindA, B, C);
    } else if (k < 64) {
        const int tile_k = (k+31)/32;
        const int n_block = (m+3)/4;
        topoCacheSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,4,1), 256*sizeof(int)>>>(
            m, k, rowindA_csr, colindA, valuesA, B, C);
    } else {
        const int tile_k = (k+63)/64;
        const int n_block = (m+8-1)/8;
        topoCacheCoarsenSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,8,1), 2*8*32*sizeof(int)>>>(
            m, k, rowindA_csr, colindA, valuesA, B, C);
    }
  }
    timer.Stop();
    time = timer.Elapsed() / EXE_NUM_CUDA;
    std::cout << "GESpMM time: " <<  time << " ms " << std::endl;
}

__global__ void LBSPMMKernel(
  int m, int k, int NNZ, int nnz_per_warp, const int* __restrict__ A_rowind, const int* __restrict__ A_colind, const  float* __restrict__ A_value, const  float* __restrict__ B,  float* __restrict__ C 
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset + threadIdx.x;
  int cid = (blockIdx.y<<5)+threadIdx.x;
  int off = blockDim.y * blockDim.x;
  int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int warp_start = warp_id * nnz_per_warp;  
  if (warp_start > NNZ - 1) {
    return;
  }
  int former_row_id = __ldg(A_rowind + warp_start);
  int current_rid = former_row_id;
  int lb = warp_start;
  int hb = warp_start + nnz_per_warp;
  if (hb > NNZ){
      hb = NNZ;
  }
  int offset;
  int ptr = lb + threadIdx.x;
  float acc1 = sum_init();
  for(int i = nnz_per_warp; i > 0; i -= 32) {
    if (ptr < hb) {
      sh[thread_idx] = __ldg(A_rowind + ptr);
      sh[thread_idx + off] = __ldg(A_colind + ptr)*k;
      reinterpret_cast<float*>(sh)[thread_idx + off + off] =  __ldg(A_value + ptr);
    } else {
      sh[thread_idx] = 0;
      sh[thread_idx + off] = 0;
      sh[thread_idx + off + off] = 0;
    }
    __syncwarp();
    #pragma unroll
    for (int kk=0; kk<32; kk++) {
      current_rid = sh[sm_offset + kk];
      if(current_rid != former_row_id) {
        atomicAdd(&C[former_row_id*k + cid], acc1);
        acc1 = sum_init();
        former_row_id = current_rid;
      }
      offset = sh[sm_offset + kk + off] + cid;
      acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+off+off)] * __ldg(B + offset));
    }
    ptr += 32;
  }
  __syncwarp();
  atomicAdd(&C[current_rid*k + cid], acc1);
}

__global__ void LBSPMMKernel_4_8_float4_float4(
  int m, int k, int NNZ, int nnz_per_warp, const int* __restrict__ A_rowind, const int* __restrict__ A_colind, const  float* __restrict__ A_value, const  float* __restrict__ B,  float* __restrict__ C 
) {
  __shared__ int row[256];
  __shared__ int col[256];
  __shared__ float val[256];
  int sm_offset = (threadIdx.y<<5);
  int thread_sh_idx = sm_offset + (threadIdx.x << 2);
  int cid = (blockIdx.y<<5)+(threadIdx.x<<2);
  int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int warp_start = warp_id * nnz_per_warp;  
  if (warp_start > NNZ - 1) {
    return;
  }
  int former_row_id = __ldg(A_rowind + warp_start);
  int current_rid = former_row_id;
  int lb = warp_start;
  int hb = warp_start + nnz_per_warp;
  if (hb > NNZ){
      hb = NNZ;
  }
  int offset;
  int ptr = lb + (threadIdx.x<<2);
  float acc1 = sum_init();
  float acc2 = sum_init();
  float acc3 = sum_init();
  float acc4 = sum_init();
  for(int i = nnz_per_warp; i > 0; i -= 32) {
    if (ptr < hb) {
      *(reinterpret_cast<int4*>(row + thread_sh_idx)) = __ldg(reinterpret_cast<const int4*>(A_rowind + ptr));
      *(reinterpret_cast<int4*>(col + thread_sh_idx)) = __ldg(reinterpret_cast<const int4*>(A_colind + ptr));
      *(reinterpret_cast<float4*>(val + thread_sh_idx))  =  __ldg(reinterpret_cast<const float4*>(A_value + ptr));
    } else {
      break;
    }
    #pragma unroll
    for (int kk=0; kk<32; kk++) {
      current_rid = row[sm_offset + kk];
      if(current_rid != former_row_id) {
        atomicAdd(&C[former_row_id*k + cid], acc1);
        atomicAdd(&C[former_row_id*k + cid + 1], acc2);
        atomicAdd(&C[former_row_id*k + cid + 2], acc3);
        atomicAdd(&C[former_row_id*k + cid + 3], acc4);
        acc1 = sum_init();
        acc2 = sum_init();
        acc3 = sum_init();
        acc4 = sum_init();
        former_row_id = current_rid;
      }
      float v = val[sm_offset+kk];
      offset = col[sm_offset + kk]*k + cid;
      float4 d = __ldg(reinterpret_cast<const float4*>(B + offset));
      acc1 = sum_reduce(acc1,  v * d.x);
      acc2 = sum_reduce(acc2, v * d.y);
      acc3 = sum_reduce(acc3,  v * d.z);
      acc4 = sum_reduce(acc4, v * d.w);
    }
    ptr += 32;
  }
  __syncwarp();
  atomicAdd(&C[current_rid*k + cid], acc1);
  atomicAdd(&C[current_rid*k + cid + 1], acc2);
  atomicAdd(&C[current_rid*k + cid + 2], acc3);
  atomicAdd(&C[current_rid*k + cid + 3], acc4);
}

#define COPY_BYTES 16
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
__global__ void LBSPMMKernel_4_8_float4_float4_async(
  int m, int k, int NNZ, int nnz_per_warp, const int* __restrict__ A_rowind, const int* __restrict__ A_colind, const  float* __restrict__ A_value, const  float* __restrict__ B,  float* __restrict__ C 
) {
  __shared__ int row[256];
  __shared__ int col[256];
  __shared__ float val[256];
  int sm_offset = (threadIdx.y<<5);
  int thread_sh_idx = sm_offset + (threadIdx.x << 2);
  int cid = (blockIdx.y<<5)+(threadIdx.x<<2);
  int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int warp_start = warp_id * nnz_per_warp;  
  if (warp_start > NNZ - 1) {
    return;
  }
  int former_row_id = __ldg(A_rowind + warp_start);
  int current_rid = former_row_id;
  int lb = warp_start;
  int hb = warp_start + nnz_per_warp;
  if (hb > NNZ){
      hb = NNZ;
  }
  int offset;
  int ptr = lb + (threadIdx.x<<2);
  float acc1 = sum_init();
  float acc2 = sum_init();
  float acc3 = sum_init();
  float acc4 = sum_init();
  uint32_t row_smem_addr = __cvta_generic_to_shared(row + thread_sh_idx);
  uint32_t col_smem_addr = __cvta_generic_to_shared(col + thread_sh_idx);
  uint32_t val_smem_addr = __cvta_generic_to_shared(val + thread_sh_idx);
  for(int i = nnz_per_warp; i > 0; i -= 32) {
    if (ptr < hb) {
      CP_ASYNC_CG(row_smem_addr, reinterpret_cast<const int4*>(A_rowind + ptr), COPY_BYTES);
      CP_ASYNC_CG(col_smem_addr, reinterpret_cast<const int4*>(A_colind + ptr), COPY_BYTES);
      CP_ASYNC_CG(val_smem_addr, reinterpret_cast<const float4*>(A_value + ptr), COPY_BYTES);
    } else {
      break;
    }
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    #pragma unroll
    for (int kk=0; kk<32; kk++) {
      current_rid = row[sm_offset + kk];
      if(current_rid != former_row_id) {
        atomicAdd(&C[former_row_id*k + cid], acc1);
        atomicAdd(&C[former_row_id*k + cid + 1], acc2);
        atomicAdd(&C[former_row_id*k + cid + 2], acc3);
        atomicAdd(&C[former_row_id*k + cid + 3], acc4);
        acc1 = sum_init();
        acc2 = sum_init();
        acc3 = sum_init();
        acc4 = sum_init();
        former_row_id = current_rid;
      }
      float v = val[sm_offset+kk];
      offset = col[sm_offset + kk]*k + cid;
      float4 d = __ldg(reinterpret_cast<const float4*>(B + offset));
      acc1 = sum_reduce(acc1,  v * d.x);
      acc2 = sum_reduce(acc2, v * d.y);
      acc3 = sum_reduce(acc3,  v * d.z);
      acc4 = sum_reduce(acc4, v * d.w);
    }
    ptr += 32;
  }
  __syncwarp();
  atomicAdd(&C[current_rid*k + cid], acc1);
  atomicAdd(&C[current_rid*k + cid + 1], acc2);
  atomicAdd(&C[current_rid*k + cid + 2], acc3);
  atomicAdd(&C[current_rid*k + cid + 3], acc4);
}

__global__ void LBSPMMKernel_4_8_float4_float4_async_double_buffer(
  int m, int k, int NNZ, int nnz_per_warp, const int* __restrict__ A_rowind, const int* __restrict__ A_colind, const  float* __restrict__ A_value, const  float* __restrict__ B,  float* __restrict__ C 
) {
  __shared__ int row[2][256];
  __shared__ int col[2][256];
  __shared__ float val[2][256];
  int sm_offset = (threadIdx.y<<5);
  int thread_sh_idx = sm_offset + (threadIdx.x << 2);
  int cid = (blockIdx.y<<5)+(threadIdx.x<<2);
  int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int warp_start = warp_id * nnz_per_warp;  
  if (warp_start > NNZ - 1) {
    return;
  }
  int former_row_id = __ldg(A_rowind + warp_start);
  int current_rid = former_row_id;
  int lb = warp_start;
  int hb = warp_start + nnz_per_warp;
  if (hb > NNZ){
      hb = NNZ;
  }
  int offset;
  int ptr = lb + (threadIdx.x<<2);
  float acc1 = sum_init();
  float acc2 = sum_init();
  float acc3 = sum_init();
  float acc4 = sum_init();
  uint32_t row_smem_addr = __cvta_generic_to_shared(row[0] + thread_sh_idx);
  uint32_t col_smem_addr = __cvta_generic_to_shared(col[0] + thread_sh_idx);
  uint32_t val_smem_addr = __cvta_generic_to_shared(val[0] + thread_sh_idx);
  CP_ASYNC_CG(row_smem_addr, reinterpret_cast<const int4*>(A_rowind + ptr), COPY_BYTES);
  CP_ASYNC_CG(col_smem_addr, reinterpret_cast<const int4*>(A_colind + ptr), COPY_BYTES);
  CP_ASYNC_CG(val_smem_addr, reinterpret_cast<const float4*>(A_value + ptr), COPY_BYTES);
  CP_ASYNC_COMMIT_GROUP();
  CP_ASYNC_WAIT_GROUP(0);
  ptr += 32;
  int tile_num = (hb-lb) / 32;
  float4 dense_matrix_fragment[32];
  for(int j = 1; j < tile_num; j++) {
    int smem_sel = (j & 1) ^ 1;
    int smem_sel_next = ( (j - 1) & 1) ^ 1;
    #pragma unroll
    for (int kk=0; kk<32; kk++) {
      offset = col[smem_sel][sm_offset + kk]*k + cid;
      dense_matrix_fragment[kk] = __ldg(reinterpret_cast<const float4*>(B + offset));
    }

    if (ptr < hb) {
      uint32_t row_smem_addr = __cvta_generic_to_shared(row[smem_sel_next] + thread_sh_idx);
      CP_ASYNC_CG(row_smem_addr, reinterpret_cast<const int4*>(A_rowind + ptr), COPY_BYTES);
      uint32_t col_smem_addr = __cvta_generic_to_shared(col[smem_sel_next] + thread_sh_idx);
      CP_ASYNC_CG(col_smem_addr, reinterpret_cast<const int4*>(A_colind + ptr), COPY_BYTES);
      uint32_t val_smem_addr = __cvta_generic_to_shared(val[smem_sel_next] + thread_sh_idx);
      CP_ASYNC_CG(val_smem_addr, reinterpret_cast<const float4*>(A_value + ptr), COPY_BYTES);
    } else {
      break;
    }
    #pragma unroll
    for (int kk=0; kk<32; kk++) {
      current_rid = row[smem_sel][sm_offset + kk];
      if(current_rid != former_row_id) {
        atomicAdd(&C[former_row_id*k + cid], acc1);
        atomicAdd(&C[former_row_id*k + cid + 1], acc2);
        atomicAdd(&C[former_row_id*k + cid + 2], acc3);
        atomicAdd(&C[former_row_id*k + cid + 3], acc4);
        acc1 = sum_init();
        acc2 = sum_init();
        acc3 = sum_init();
        acc4 = sum_init();
        former_row_id = current_rid;
      }
      float v = val[smem_sel][sm_offset+kk];
      acc1 = sum_reduce(acc1,  v * dense_matrix_fragment[kk].x);
      acc2 = sum_reduce(acc2, v * dense_matrix_fragment[kk].y);
      acc3 = sum_reduce(acc3,  v * dense_matrix_fragment[kk].z);
      acc4 = sum_reduce(acc4, v * dense_matrix_fragment[kk].w);
    }
    ptr += 32;
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
  }
  int smem_sel = (tile_num & 1) ^ 1;
  #pragma unroll
  for (int kk=0; kk<32; kk++) {
    current_rid = row[smem_sel][sm_offset + kk];
    if(current_rid != former_row_id) {
      atomicAdd(&C[former_row_id*k + cid], acc1);
      atomicAdd(&C[former_row_id*k + cid + 1], acc2);
      atomicAdd(&C[former_row_id*k + cid + 2], acc3);
      atomicAdd(&C[former_row_id*k + cid + 3], acc4);
      acc1 = sum_init();
      acc2 = sum_init();
      acc3 = sum_init();
      acc4 = sum_init();
      former_row_id = current_rid;
    }
    float v = val[smem_sel][sm_offset+kk];
    offset = col[smem_sel][sm_offset + kk]*k + cid;
    float4 d = __ldg(reinterpret_cast<const float4*>(B + offset));
    acc1 = sum_reduce(acc1,  v * d.x);
    acc2 = sum_reduce(acc2, v * d.y);
    acc3 = sum_reduce(acc3,  v * d.z);
    acc4 = sum_reduce(acc4, v * d.w);
  }  
  atomicAdd(&C[current_rid*k + cid], acc1);
  atomicAdd(&C[current_rid*k + cid + 1], acc2);
  atomicAdd(&C[current_rid*k + cid + 2], acc3);
  atomicAdd(&C[current_rid*k + cid + 3], acc4);
}

void LBSpMM(const COO_Matrix &mat1, const Matrix &mat2, Matrix &result, float& time) {
  GpuTimer timer;
  const int n_rows_mat1 = mat1.num_rows();
  const int n_cols_mat1 = mat1.num_cols();
  const int n_rows_mat2 = mat2.dim(0);
  const int n_cols_mat2 = mat2.dim(1);
  const int m = n_rows_mat1;
  const int k = n_cols_mat2;
  using scalar_t = float;
  const int nnzA = mat1.values().size();
  const int *rowindA = mat1.row_indices().data<int>();
  const int *colindA = mat1.column_indices().data<int>();
  const scalar_t *valuesA = mat1.values().data<scalar_t>();
  scalar_t *B = mat2.mutable_data<scalar_t>();
  scalar_t *C = result.mutable_data<scalar_t>();
  const int tile_k = (k+31)/32;
  int nnz_per_warp = 32;
  if ( (nnzA / m > 256) && (m > 5000)) nnz_per_warp = 128; // 256
  const int n_block = (((nnzA+nnz_per_warp-1)/nnz_per_warp + 7) / 8);
  timer.Start();
  for (int i = 0; i < EXE_NUM_CUDA; i++) {
    if(nnz_per_warp <= 32)
      LBSPMMKernel<<< dim3(n_block, tile_k, 1), dim3(32,8,1), 768*sizeof(int)>>>(
      m, k, nnzA, nnz_per_warp, rowindA, colindA, valuesA, B, C);
      // LBSPMMKernel_4_8_float4_float4<<< dim3(n_block, tile_k, 1), dim3(8,8,1)>>>(
      //     m, k, nnzA, nnz_per_warp, rowindA, colindA, valuesA, B, C);
      // LBSPMMKernel_4_8_float4_float4_async<<< dim3(n_block, tile_k, 1), dim3(8,8,1)>>>(
      //       m, k, nnzA, nnz_per_warp, rowindA, colindA, valuesA, B, C);
    else
      LBSPMMKernel_4_8_float4_float4_async_double_buffer<<< dim3(n_block, tile_k, 1), dim3(8,8,1)>>>(
              m, k, nnzA, nnz_per_warp, rowindA, colindA, valuesA, B, C);
  }
  timer.Stop();
  time = timer.Elapsed() / EXE_NUM_CUDA;
  std::cout << "HPSpMM time: " <<  time << " ms " << std::endl;
}





