// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
__global__ void kernel_get_lab_val(const int num, const int dim,
    const Dtype* label, const Dtype* margin, Dtype* data, Dtype* val) {
  CUDA_KERNEL_LOOP(index, num) {
    val[index] = data[index * dim + static_cast<int>(label[index])];
    // the loss for labeled data is zero
    data[index * dim + static_cast<int>(label[index])] -= *margin;
  }
}

template <typename Dtype>
__global__ void kernel_hinge_max(const int num, const int dim,
    const Dtype* labelval, const Dtype* margin, Dtype* data) {
  CUDA_KERNEL_LOOP(index, num * dim){
    int n = index / dim;
    data[index] = max(Dtype(0), *margin - labelval[n] + data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_set_diff(const int num, const int dim,
    const Dtype* label, const Dtype* val, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, num) {
    diff[index * dim + static_cast<int>(label[index])] = val[index];
  }
}

template <typename Dtype>
Dtype HingeRankLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* labval_data = labval_.mutable_gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* margin_data = margin_.gpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  caffe_gpu_copy(count, bottom_data, bottom_diff);
  kernel_get_lab_val<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
      num, dim, label, margin_data, bottom_diff, labval_data);
  kernel_hinge_max<Dtype><<<CAFFE_GET_BLOCKS(num * dim),
    CAFFE_CUDA_NUM_THREADS>>>(num, dim, labval_data, margin_data, bottom_diff);
  Dtype sum;
  caffe_gpu_asum(count, bottom_diff, &sum);
  return  sum / num;
}

template <typename Dtype>
void HingeRankLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const Dtype* label = (*bottom)[1]->gpu_data();
  Dtype* labval_data = labval_.mutable_gpu_data();
  int num = (*bottom)[0]->num();
  int count = (*bottom)[0]->count();
  int dim = count / num;

  caffe_gpu_sign(count, bottom_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, -1., bottom_diff,
      sum_multiplier_.gpu_data(), 0., labval_data);
  kernel_set_diff<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
      num, dim, label, labval_data, bottom_diff);
  caffe_gpu_scal(count, Dtype(1. / num), bottom_diff);
}

INSTANTIATE_CLASS(HingeRankLossLayer);


}  // namespace caffe
