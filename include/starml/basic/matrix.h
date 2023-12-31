#pragma once
#include <vector>

#include "starml/basic/allocator.h"
#include "starml/basic/device.h"
#include "starml/basic/type.h"
#include "starml/basic/scalar.h"
#include "starml/utils/loguru.h"
#include "starml/basic/handle.h"

namespace starml {
using Shape = std::vector<int>;

class Matrix {
 public:
  Matrix();
  Matrix(const Shape& shape, const Device& device, const DataType& data_type);
  Matrix(const Shape& shape, const DataType& data_type, const Device& device);
  ~Matrix() = default;
  Matrix(const Matrix& rhs) = default;
  Matrix& operator=(const Matrix& rhs) = default;

  // Get total number of elements in the matrix
  int size() const;
  // Get the shape of matrix
  const Shape& dims() const;
  // Get the number of element in a specific axis
  int dim(int axis) const;
  // Get the number of dimension of matrix
  int ndims() const;
  // Get the device message
  const Device& device() const;
  // Get the data type message
  const DataType& data_type() const;

  bool is_cuda() const;

  // Get the raw data pointer (void *)
  const void* raw_data() const;
  void* raw_mutable_data() const;

  // Transfer the matrix to specific device
  Matrix to(Device new_device, Handle* handle = NULL) const;
  // Print the matrix for debug usage
  void print(std::string file_name = "") const;

  // The rules to judge whether the input data type is valid:
  // 1. int can not convert to float or double, vice versa 
  // 2. data can be casted when the bytes of the input data type 
  // is equal to the bytes of data_ptr_ (The rule is mainly for int, short, long 
  // , since the bytes of int may be variance on different system)
  template <typename T>
  const T* data() const {
    STARML_CHECK(dtype_.is_valid<T>())
        << "Input template data type is not valid since the data type for "
           "matrix is "
        << dtype_.type();
    return static_cast<T*>(data_ptr_.get());
  }
  template <typename T>
  T* mutable_data() const {
    STARML_CHECK(dtype_.is_valid<T>())
        << "Input template data type is not valid since the data type for "
           "matrix is "
        << dtype_.type();
    return static_cast<T*>(data_ptr_.get());
  }

 private:
  void initial();
  int size_;
  Device device_;
  DataType dtype_;
  Allocator* allocator_;
  DataPtr data_ptr_;
  Shape shape_;
};
}  // namespace starml