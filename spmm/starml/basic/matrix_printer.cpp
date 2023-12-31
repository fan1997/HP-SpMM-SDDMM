#include "starml/basic/matrix_printer.h"
#include "starml/utils/loguru.h"
#include "starml/basic/dispatch.h"

namespace starml {
MatrixPrinter::MatrixPrinter(const std::string& file_name, int limit)
    : to_file_(!file_name.empty()), limit_(limit ? limit : k_limit_default) {
  if (to_file_) {
    // initialize the unique_ptr, delete the output file with the
    // same name and create a new one.
    log_file_.reset(new std::ofstream(
        file_name, std::ofstream::out | std::ofstream::trunc));
    STARML_CHECK(log_file_->good())
        << "Failed to open MatrixPrinter file " << file_name
        << ". rdstate() = " << log_file_->rdstate();
  }
}

MatrixPrinter::~MatrixPrinter() {
  if (log_file_.get()) {
    log_file_->close();
  }
}

void MatrixPrinter::print(const Matrix& matrix) {
  auto data_type = matrix.data_type().type();
  STARML_DISPATCH_TYPES(data_type, "matrix printer", [&] {
    if (matrix.ndims() == 2) {
      print_matrix<scalar_t>(matrix);
    } else {
      print<scalar_t>(matrix);
    }
  });
}

std::string MatrixPrinter::meta_string(const Matrix& matrix) {
  std::stringstream meta_stream;
  meta_stream << "Matrix "  << " of type "
              << matrix.data_type().type() << " on "
              << matrix.device().type() << ":\n\tdims: ";
  auto dims = matrix.dims();
  for (int i = 0; i < dims.size(); ++i) {
    meta_stream << dims[i];
    if (i + 1 != dims.size()) {
      meta_stream << ",";
    }
  }
  return meta_stream.str();
}
}  // namespace starml