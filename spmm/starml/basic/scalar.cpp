#include "starml/basic/scalar.h"
#include "starml/basic/dispatch.h"
#include "starml/operators/factories.h"

namespace starml {
Scalar::Scalar(int8_t v) : type_(kInt8) { value_.ival8_ = v; }
Scalar::Scalar(int16_t v) : type_(kInt16) { value_.ival16_ = v; }
Scalar::Scalar(int32_t v) : type_(kInt32) { value_.ival32_ = v; }
Scalar::Scalar(int64_t v) : type_(kInt64) { value_.ival64_ = v; }
Scalar::Scalar(float v) : type_(kFloat) { value_.fval_ = v; }
Scalar::Scalar(double v) : type_(kDouble) { value_.dval_ = v; }

bool Scalar::operator==(const Scalar& rhs) {
  auto data_type = (this->type_ < rhs.type()) ? rhs.type() : this->type_;
  STARML_DISPATCH_TYPES(data_type, "SCALAR_CMP", [&]() {
    return value<scalar_t>() == rhs.value<scalar_t>();
  });
}

bool Scalar::operator!=(const Scalar& rhs) { return !((*this) == rhs); }

bool Scalar::operator>=(const Scalar& rhs) {
  auto data_type = (this->type_ < rhs.type()) ? rhs.type() : this->type_;
  STARML_DISPATCH_TYPES(data_type, "SCALAR_CMP", [&]() {
    return value<scalar_t>() >= rhs.value<scalar_t>();
  });
}

bool Scalar::operator>(const Scalar& rhs) {
  auto data_type = (this->type_ < rhs.type()) ? rhs.type() : this->type_;
  STARML_DISPATCH_TYPES(data_type, "SCALAR_CMP", [&]() {
    return value<scalar_t>() > rhs.value<scalar_t>();
  });
}

bool Scalar::operator<(const Scalar& rhs) { return !((*this) >= rhs); }

bool Scalar::operator<=(const Scalar& rhs) { return !((*this) > rhs); }

DataTypeKind Scalar::type() const { return type_; }

Matrix Scalar::to_matrix(const Device& device, const DataType& data_type) const{
  return full({1}, device, data_type, *this);
}

}  // namespace starml