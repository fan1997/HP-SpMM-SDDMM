#include "spmm/utils/compare.h"
void ExpectEqualDenseWithError(const Matrix& mat1, const Matrix& mat2, std::string ktype) {
  STARML_CHECK_EQ(mat1.size(), mat2.size());
  STARML_CHECK_EQ(mat1.data_type().type(), mat2.data_type().type());
  STARML_CHECK_EQ(mat1.ndims(), mat2.ndims());
  Matrix mat1_new =  mat1.to(kCPU);
  Matrix mat2_new = mat2.to(kCPU);
  auto size = mat1_new.size();
  float err_max = 0, err = 0;
  int pos = -1;
  for(int i = 0; i < size; ++i) {
    if(ktype == "div"){
      err = std::fabs(mat1_new.data<float>()[i] / (float)(EXE_NUM_CUDA) - mat2_new.data<float>()[i]);
    }
    else{
      err = std::fabs(mat1_new.data<float>()[i] - mat2_new.data<float>()[i]);
    }
    err_max = std::max(err_max, err);
    pos = i;
  }
  bool passed = err_max < 0.1;
  if (!passed) {
    std::cout<<"Wrong!! max_err = "<< err_max << " > 0.1 " << "at pos: " << pos << std::endl;
  }else{
    std::cout<<"PASSed!! max_err = "<< err_max << " < 0.1 "<< std::endl;
  }
  STARML_CHECK(passed);
}

void ExpectEqualDenseWithErrorTrans(const Matrix& mat1, const Matrix& mat2, std::string ktype) {
  STARML_CHECK_EQ(mat1.size(), mat2.size());
  STARML_CHECK_EQ(mat1.data_type().type(), mat2.data_type().type());
  STARML_CHECK_EQ(mat1.ndims(), mat2.ndims());
  Matrix mat1_new =  mat1.to(kCPU);
  Matrix mat2_new = mat2.to(kCPU);
  auto size = mat1_new.size();
  auto num_rows = mat1_new.dim(0);
  auto num_cols = mat1_new.dim(1);
  float err_max = 0, err = 0, err_rs = 0;
  for(int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++){
      if(ktype == "div"){
        err = std::fabs(mat1_new.data<float>()[i * num_cols + j] / (float)(EXE_NUM_CUDA) - mat2_new.data<float>()[j * num_rows + i]);
        err_rs  = std::fabs(mat1_new.data<float>()[i * num_cols + j]- mat2_new.data<float>()[j * num_rows + i]);
        err = std::min(err_rs, err); 
      } else {
        err = std::fabs(mat1_new.data<float>()[i * num_cols + j] - mat2_new.data<float>()[j * num_rows + i]);
      }
      err_max = std::max(err_max, err); 
    }
  }
  bool passed = err_max < 0.1;
  if (!passed) {
    std::cout<<"Wrong!! max_err = "<< err_max << " > 0.1 "<< std::endl;
  }else{
    std::cout<<"PASSed!! max_err = "<< err_max << " < 0.1 "<< std::endl;
  }
  STARML_CHECK(passed);
}



