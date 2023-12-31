#pragma once
#include "starml/basic/type.h"
#include "starml/basic/device.h"
namespace starml {
class Matrix;
// This class uses to represent Scalar values, scalar means 0-dimension matrix
// it is helpful while initialize all the elements in the matrix as some special
// value. The class can be initialized both in implicit or explicit ways.
class Scalar {
 public:
  Scalar(int8_t v);
  Scalar(int16_t v);
  Scalar(int32_t v);
  Scalar(int64_t v);
  Scalar(float v);
  Scalar(double v);
  ~Scalar() = default;
  Scalar(const Scalar& rhs) = default;
  Scalar& operator=(const Scalar& rhs) = default;

  // Overload operators
  bool operator==(const Scalar& rhs);
  bool operator!=(const Scalar& rhs);
  bool operator<(const Scalar& rhs);
  bool operator>(const Scalar& rhs);
  bool operator<=(const Scalar& rhs);
  bool operator>=(const Scalar& rhs);

  // Get the data type it represents currently
  DataTypeKind type() const;

  // Get the value stores in object which cast to given data type
  template <typename T>
  T value() const {
    if (kInt8 == type_) {
      return static_cast<T>(value_.ival8_);
    } else if (kInt16 == type_) {
      return static_cast<T>(value_.ival16_);
    } else if (kInt32 == type_) {
      return static_cast<T>(value_.ival32_);
    } else if (kInt64 == type_) {
      return static_cast<T>(value_.ival64_);
    } else if (kFloat == type_) {
      return static_cast<T>(value_.fval_);
    } else {
      return static_cast<T>(value_.dval_);
    }
  }

  Matrix to_matrix(const Device& device_type, const DataType& data_type) const;

 private:
  // Represent the current data_type store in the object.
  DataTypeKind type_;
  // All data store in union value_ share the same memory.
  union {
    int8_t ival8_;
    int16_t ival16_;
    int32_t ival32_;
    int64_t ival64_;
    float fval_;
    double dval_;
  } value_;
};
}  // namespace starml