#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
#include "starml/basic/matrix_printer.h"
#include "starml/operators/factories.h"
#include "starml/operators/copy.h"

namespace starml {

void Matrix::initial() {
  this->size_ = 1;
  for (auto item : shape_) {
    this->size_ *= item;
  }
  this->allocator_ = get_allocator(device_.type());
  this->data_ptr_ = allocator_->allocate(size_ * dtype_.size());
}

Matrix::Matrix() : allocator_(NULL), data_ptr_(NULL) {}
Matrix::Matrix(const Shape& shape, const Device& device,
               const DataType& data_type)
    : shape_(shape), device_(device), dtype_(data_type) {
  initial();
}
Matrix::Matrix(const Shape& shape, const DataType& data_type,
               const Device& device)
    : shape_(shape), dtype_(data_type), device_(device) {
  initial();
}

int Matrix::size() const { return this->size_; }
int Matrix::dim(int axis) const { return (dims())[axis]; }
const Shape& Matrix::dims() const { return this->shape_; }
int Matrix::ndims() const { return this->shape_.size(); }
const Device& Matrix::device() const { return this->device_; }
const DataType& Matrix::data_type() const { return this->dtype_; }

const void* Matrix::raw_data() const { return this->data_ptr_.get(); }
void* Matrix::raw_mutable_data() const { return this->data_ptr_.get(); }

bool Matrix::is_cuda() const { return device_.type() == kCUDA; }

Matrix Matrix::to(Device new_device, Handle* handle) const {
  return deep_copy(*this, new_device, handle);
}

void Matrix::print(std::string file_name) const {
  MatrixPrinter mp(file_name);
  mp.print(*this);
}

}  // namespace starml