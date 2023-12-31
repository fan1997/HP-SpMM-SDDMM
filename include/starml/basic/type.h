#pragma once
#include <cxxabi.h>
#include <cstddef>
#include <ostream>
#include <string>
#include <typeinfo>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <functional>
namespace starml {
// Supported data type, uncertain represents the data type is not sepecific.
// Matrix with UNCERTAIN data type may cause an error in further use.
enum class DataTypeKind : int {
  UNCERTAIN = -1,
  Int8 = 0,
  Int16 = 1,
  Int32 = 2,
  Int64 = 3,
  Float = 4,
  Double = 5,
  Half = 6
};
// The constexpr specifier declares that it is possible to evaluate
// the value of the variable at compile time. Since the value of enum
// will be known during compiling time
constexpr DataTypeKind kInt8 = DataTypeKind::Int8;
constexpr DataTypeKind kInt16 = DataTypeKind::Int16;
constexpr DataTypeKind kInt32 = DataTypeKind::Int32;
constexpr DataTypeKind kInt64 = DataTypeKind::Int64;
constexpr DataTypeKind kFloat = DataTypeKind::Float;
constexpr DataTypeKind kDouble = DataTypeKind::Double;
constexpr DataTypeKind kHalf = DataTypeKind::Half;
// In order to print or return the right data type string representation
// lower_case is a flag which indicates the spelling style
// of the string
std::string to_string(DataTypeKind type, bool lower_case);
std::ostream& operator<<(std::ostream& stream, DataTypeKind type);

// Maintain the attributes relative to the Data Type
class DataType {
 public:
  // Default DataTypeKind is UNCERTAIN, which may cause error in further use
  DataType(DataTypeKind type = DataTypeKind::UNCERTAIN);
  ~DataType() = default;
  DataType(const DataType& rhs) = default;
  DataType& operator=(const DataType& rhs) = default;
  // Return the bytes size relative to the DataTypeKind
  size_t size() const;
  // Return the specific DataTypeKind
  DataTypeKind type() const;
  // Using RTTI to get the string name of data type
  std::string name() const;
  bool operator==(const DataType& rhs) const;
  bool operator!=(const DataType& rhs) const;

  bool is_int() const;

  // Template judgement whether the given data type is consistent with
  // the type object contains.
  template <typename T>
  bool is_valid() const {
    if (name() == type_name<T>()) {
      return true;
    }
    return false;
  }

 private:
  // RTTI convert the input data type to string
  template <typename T>
  static std::string type_name();
  DataTypeKind type_;
  // Hash table, given the relation between DataTypeKind and corresponding byte
  // size.
  static std::unordered_map<int, size_t> type_sizes;
};

template <typename T>
std::string DataType::type_name() {
  const char* name = typeid(T).name();
  int status = -1;
  std::unique_ptr<char, std::function<void(char*)>> demangled(
      abi::__cxa_demangle(name, nullptr, 0, &status), free);
  if (status == 0) {
    return demangled.get();
  } else {
    return name;
  }
}


}  // namespace starml