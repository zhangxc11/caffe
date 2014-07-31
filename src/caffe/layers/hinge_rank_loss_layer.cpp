// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void HingeRankLossLayer<Dtype>::FurtherSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  margin_ = this->layer_param_.hinge_rank_loss_param().margin();
}

template <typename Dtype>
Dtype HingeRankLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype labval;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    labval = bottom_diff[i * dim + static_cast<int>(label[i])];
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] =
        max(Dtype(0), margin_ - labval + bottom_diff[i * dim + j]);
    }
  }
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] = 0;
  }
  return caffe_cpu_asum(count, bottom_diff) / num;
}

template <typename Dtype>
void HingeRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = (*bottom)[0]->num();
  int count = (*bottom)[0]->count();
  int dim = count / num;

  caffe_cpu_sign(count, bottom_diff, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] =
      - caffe_cpu_asum(dim, bottom_diff + i * dim);
  }
  caffe_scal(count, Dtype(1. / num), bottom_diff);
}

INSTANTIATE_CLASS(HingeRankLossLayer);

}  // namespace caffe
