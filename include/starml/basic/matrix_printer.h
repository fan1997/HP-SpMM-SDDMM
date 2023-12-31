#pragma once
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include "starml/basic/matrix.h"

namespace starml {
// Maximum limitation of matrix elements to print
constexpr int k_limit_default = 180;

// Helper class, print matrix in consistent style.
// It is recommend to use one MatrixPrinter instance corresponding to
// exactly one Matrix instance. The content of this matrix will be printed
// on the console (without filename supply) or written to a specific file.
// Note: Copy constructor/assignment is not supported here due to the unique_ptr
class MatrixPrinter {
 public:
  MatrixPrinter(const std::string& file_name = "", int limit = k_limit_default);
  // Deconstructor is needed to close the file manually
  ~MatrixPrinter();

  // Sepecific print function, std::stringstream makes format and concat easier 
  void print(const Matrix& matrix);
  template <typename T>
  void print(const Matrix& matrix);
  template <typename T>
  void print_matrix(const Matrix& matrix, int n_row_limited = 20,
                    int n_col_limited = 20);

  // Return the description string message of the matrix
  // including name, dtype, device and etc.
  std::string meta_string(const Matrix& matrix);

 private:
  bool to_file_;
  int limit_;
  std::unique_ptr<std::ofstream> log_file_;
};

template <typename T>
void MatrixPrinter::print(const Matrix& matrix) {
  std::stringstream values_stream;
  int total_count = std::min(matrix.size(), limit_);
  Matrix host_matrix = matrix.device().is_cpu() ? matrix : matrix.to(kCPU);
  const T* data = host_matrix.data<T>();
  values_stream << "\n\t"
                << "data: ";
  for (int i = 0; i < total_count; ++i) {
    values_stream << data[i] << ", ";
  }
  if (to_file_) {
    (*log_file_) << meta_string(matrix) << values_stream.str() << std::endl;
  } else {
    std::cout << meta_string(matrix) << values_stream.str() << std::endl;
  }
}

template <typename T>
void MatrixPrinter::print_matrix(const Matrix& matrix, int n_row_limited,
                                 int n_col_limited) {
  std::stringstream values_stream;
  Matrix host_matrix = matrix.device().is_cpu() ? matrix : matrix.to(kCPU);

  auto n_row = std::min(matrix.dim(0), n_row_limited);
  auto n_col = std::min(matrix.dim(1), n_col_limited);
  const T* data = host_matrix.data<T>();
  values_stream << "\n\t"
                << "data: ";
  for (int i = 0; i < n_row; ++i) {
    for (int j = 0; j < n_col; ++j) {
      values_stream << data[i * n_col + j];
      if (j + 1 != n_col) {
        values_stream << ",\t"; 
      }
    }
    if (i != n_row - 1) {
      values_stream << "\n\t"
                    << "      ";
    }
  }
  if (to_file_) {
    (*log_file_) << meta_string(matrix) << values_stream.str() << std::endl;
  } else {
    std::cout << meta_string(matrix) << values_stream.str() << std::endl;
  }
}
}  // namespace starml