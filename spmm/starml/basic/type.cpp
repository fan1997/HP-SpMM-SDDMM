#include "starml/basic/type.h"

#include <stdint.h>

#include "starml/basic/dispatch.h"
#include "starml/utils/loguru.h"

namespace starml {
std::string to_string(DataTypeKind d, bool lower_case) {
  switch (d) {
    case DataTypeKind::Int8:
      return lower_case ? "int8" : "INT8";
    case DataTypeKind::Int16:
      return lower_case ? "int16" : "INT16";
    case DataTypeKind::Int32:
      return lower_case ? "int32" : "INT32";
    case DataTypeKind::Int64:
      return lower_case ? "int64" : "INT64";
    case DataTypeKind::Float:
      return lower_case ? "float" : "FLOAT";
    case DataTypeKind::Double:
      return lower_case ? "double" : "DOUBLE";
    case DataTypeKind::Half:
      return lower_case ? "half" : "HALF";
    default:
      STARML_LOG(ERROR) << "Unknown data type: " << static_cast<int>(d);
      return "";
  }
}

std::ostream& operator<<(std::ostream& os, DataTypeKind type) {
  os << to_string(type, true);
  return os;
}

std::unordered_map<int, size_t> DataType::type_sizes{
    {0, sizeof(int8_t)},  {1, sizeof(int16_t)}, {2, sizeof(int32_t)},
    {3, sizeof(int64_t)}, {4, sizeof(float)},   {5, sizeof(double)}};

DataType::DataType(DataTypeKind type) : type_(type) {}

size_t DataType::size() const {
  STARML_CHECK_NE(static_cast<int>(this->type_), -1)
      << "Data type is uncertain.";
  return type_sizes[static_cast<int>(this->type_)];
}
DataTypeKind DataType::type() const {
  STARML_LOG_IF(WARNING, (static_cast<int>(type_) == -1))
      << "Data type is uncertain.";
  return this->type_;
}

std::string DataType::name() const {
  std::string result = "";
  STARML_DISPATCH_TYPES(type_, "test",
                        [&]() { result = type_name<scalar_t>(); });
  return result;
}

bool DataType::operator==(const DataType& rhs) const {
  return this->type_ == rhs.type_;
}
bool DataType::operator!=(const DataType& rhs) const {
  return !((*this) == rhs);
}

bool DataType::is_int() const {
  if (this->type_ == kInt8 || this->type_ == kInt16 || this->type_ == kInt32 ||
      this->type_ == kInt64) {
    return true;
  }
  return false;
}

}  // namespace starml