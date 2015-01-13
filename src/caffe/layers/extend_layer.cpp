#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ExtendLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& ,
      vector<Blob<Dtype>*>* ) {
  const ExtendParameter& extend_param = this->layer_param_.extend_param();
  CHECK_EQ(extend_param.pos_size(), extend_param.dim_size()) << "size of dim and pos must be equal";
  extend_pos_.clear();
  extend_dim_.clear();
  std::copy(extend_param.pos().begin(),
      extend_param.pos().end(),
      std::back_inserter(extend_pos_));
  std::copy(extend_param.dim().begin(),
      extend_param.dim().end(),
      std::back_inserter(extend_dim_));
}

template <typename Dtype>
void ExtendLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  count_ = 0;
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(extend_pos_.size(), top->size());
  for (unsigned i = 0; i < top->size(); ++i) {
    (*top)[i]->Reshape(num_, 1, 1, extend_dim_[i]);
  }
}

template <typename Dtype>
void ExtendLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  for (unsigned i = 0; i < top->size(); ++i) {
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    for (int k = 0; k < extend_dim_[i]; k++) {
      top_data[k] = 0;
    }
    top_data[int(bottom_data[extend_pos_[i]])] = 1;
  }
}

#ifdef CPU_ONLY
// STUB_GPU(ExtendLayer);
#endif

INSTANTIATE_CLASS(ExtendLayer);

}  // namespace caffe
