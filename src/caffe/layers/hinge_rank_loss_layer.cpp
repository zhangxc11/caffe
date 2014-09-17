
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void HingeRankLossLayer<Dtype>::FurtherSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  margin_.Reshape(1, 1, 1, 1);
  Dtype* margin_data = margin_.mutable_cpu_data();
  *margin_data = this->layer_param_.hinge_rank_loss_param().margin();
  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }
  labval_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
Dtype HingeRankLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* labval_data = labval_.mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* margin_data = margin_.cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    labval_data[i] = bottom_diff[i * dim + static_cast<int>(label[i])];
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] =
        max(Dtype(0), *margin_data - labval_data[i] + bottom_diff[i * dim + j]);
    }
  }
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] = 0;
  }
  return caffe_cpu_asum(count, bottom_diff) / num;
}

template <typename Dtype>
void HingeRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* label = (*bottom)[1]->cpu_data();
  Dtype* labval_data = labval_.mutable_cpu_data();
  int num = (*bottom)[0]->num();
  int count = (*bottom)[0]->count();
  int dim = count / num;

  caffe_cpu_sign(count, bottom_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, -1., bottom_diff,
      sum_multiplier_.cpu_data(), 0., labval_data);
  for (int i = 0; i < num; ++i) {
    // bottom_diff[i * dim + static_cast<int>(label[i])] =
    //  - caffe_cpu_asum(dim, bottom_diff + i * dim);
    bottom_diff[i * dim + static_cast<int>(label[i])] = labval_data[i];
  }
  caffe_scal(count, Dtype(1. / num), bottom_diff);
}

INSTANTIATE_CLASS(HingeRankLossLayer);

}  // namespace caffe
