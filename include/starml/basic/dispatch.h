#pragma once
#include <mutex>
#include <typeinfo>
#include <unordered_map>
#include <initializer_list>
#include "starml/basic/matrix.h"
#include "starml/utils/macros.h"

namespace starml {

template <typename TFnPtr>
class Dispatcher;

template <typename TReturn, typename... TArgs>
class Dispatcher<TReturn (*)(TArgs...)> {
 public:
  using FnPtr = TReturn (*)(TArgs...);
  template <typename... TArgTypes>
  TReturn operator()(TArgTypes&&... args) {
    int key = dispatch_key(args...);
    FnPtr kernel = kernel_table_[key];
    return (*kernel)(std::forward<TArgTypes>(args)...);
  }
  void set_dispatcher(std::vector<DeviceType> device_types, FnPtr kernel) {
    std::lock_guard<std::mutex> guard(mu_);
    int key = 0;
    for (int i = device_types.size() - 1; i >= 0; i--) {
      key = kNumDeviceTypes * key + static_cast<int>(device_types[i]);
    }
    kernel_table_[key] = kernel;
  }

 protected:
  template <typename T>
  int dispatch_key(const T& arg) {
    return matrix_type_id(arg);
  }

  template <typename THead, typename... TTail>
  int dispatch_key(const THead& head, const TTail&... tail) {
    if (typeid(head) == typeid(Matrix)) {
      return matrix_type_id(head) + kNumDeviceTypes * dispatch_key(tail...);
    }
    return dispatch_key(tail...);
  }

  int matrix_type_id(const Matrix& matrix) {
    return static_cast<int>(matrix.device().type());
  }

  template <typename T>
  int matrix_type_id(const T& non_matrix) {
    return 0;
  }

  std::mutex mu_;
  std::unordered_map<int, FnPtr> kernel_table_;
};

template <typename Obj, typename FnPtr>
class DispatcherRegister {
 public:
  DispatcherRegister(std::vector<DeviceType> device_types, FnPtr kernel) {
    Obj::singleton().set_dispatcher(device_types, kernel);
  }
};

#define STARML_DECLARE_DISPATCHER(dispatcher, kernel_fn_type)  \
  class dispatcher##_t : public Dispatcher<kernel_fn_type> {   \
   public:                                                     \
    static dispatcher##_t& singleton() {                       \
      static dispatcher##_t dispatcher;                        \
      return dispatcher;                                       \
    }                                                          \
                                                               \
   private:                                                    \
    dispatcher##_t() {}                                        \
    dispatcher##_t(const dispatcher##_t&) = delete;            \
    dispatcher##_t& operator=(dispatcher##_t const&) = delete; \
  };                                                           \
  extern dispatcher##_t& dispatcher

#define STARML_DEFINE_DISPATCHER(dispatcher) \
  dispatcher##_t& dispatcher = dispatcher##_t::singleton()

#define STARML_REGISTER_KERNEL(dispatcher, fn, ...)       \
  static DispatcherRegister<dispatcher##_t, decltype(fn)> \
      STARML_ANONYMOUS_VARIABLE(register##dispatcher)(    \
          std::initializer_list<DeviceType>{__VA_ARGS__}, fn)

#define STARML_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                    \
    using scalar_t = type;                             \
    return __VA_ARGS__();                              \
  }

#define STARML_DISPATCH_FLOATING_TYPES(SCALAR_TYPE, NAME, ...) \
  [&] {                                                        \
    switch (SCALAR_TYPE) {                                     \
      STARML_PRIVATE_CASE_TYPE(kDouble, double, __VA_ARGS__)   \
      STARML_PRIVATE_CASE_TYPE(kFloat, float, __VA_ARGS__)     \
      default:                                                 \
        break;                                                 \
    }                                                          \
  }()

#define STARML_DISPATCH_TYPES(SCALAR_TYPE, NAME, ...)                         \
  [&] {                                                                       \
    switch (SCALAR_TYPE) {                                                    \
      STARML_PRIVATE_CASE_TYPE(kInt8, int8_t, __VA_ARGS__)                    \
      STARML_PRIVATE_CASE_TYPE(kInt16, int16_t, __VA_ARGS__)                  \
      STARML_PRIVATE_CASE_TYPE(kInt32, int32_t, __VA_ARGS__)                  \
      STARML_PRIVATE_CASE_TYPE(kInt64, int64_t, __VA_ARGS__)                  \
      STARML_PRIVATE_CASE_TYPE(kDouble, double, __VA_ARGS__)                  \
      STARML_PRIVATE_CASE_TYPE(kFloat, float, __VA_ARGS__)                    \
      default:                                                                \
        STARML_LOG(ERROR) << #NAME << " not implemented for '" << SCALAR_TYPE \
                          << "'";                                             \
    }                                                                         \
  }()

}  // namespace starml