#pragma once

namespace starml {
#define STARML_CONCAT_IMPL(x, y) x##y
#define STARML_MACRO_CONCAT(x, y) STARML_CONCAT_IMPL(x, y)
#define STARML_MACRO_EXPAND(args) args

#ifdef __COUNTER__
#define STARML_ANONYMOUS_VARIABLE(str) STARML_MACRO_CONCAT(str, __COUNTER__)
#else
#define STARML_ANONYMOUS_VARIABLE(str) STARML_MACRO_CONCAT(str, __LINE__)
#endif
}  // namespace starml