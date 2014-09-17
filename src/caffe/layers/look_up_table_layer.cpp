#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void LookUpTableLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1)
    << "Look Up Table Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1)
    << "Look Up Table Layer takes a single blob as output.";

  BlobProto blob_proto;
  ReadProtoFromBinaryFile(
    this->layer_param_.look_up_table_param().source(),
    &blob_proto);
  dictionary_.FromProto(blob_proto);
  // Figure out the dimensions
  const int num_output = dictionary_.width();
  N_ = num_output;
  CHECK_EQ(dictionary_.num(), 1);
  CHECK_EQ(dictionary_.channels(), 1);

  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
}

template <typename Dtype>
Dtype LookUpTableLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* diction_data = this->dictionary_.cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  for (int i = 0; i < bottom[0]->num(); i++) {
    caffe_copy(N_, diction_data + static_cast<int>(bottom_data[i]) * N_,
        top_data + i * N_);
  }
  return Dtype(0);
}

template <typename Dtype>
void LookUpTableLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  return;
}

INSTANTIATE_CLASS(LookUpTableLayer);


}  // namespace caffe
