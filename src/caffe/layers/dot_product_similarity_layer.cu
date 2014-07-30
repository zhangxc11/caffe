// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
__global__ void kernel_get_max(const int num, const int dim,
    const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype maxval = -FLT_MAX;
    for (int i = 0; i < dim; ++i) {
      maxval = max(data[index * dim + i], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_dotp_div(const int num, const int dim,
    const Dtype* scale, Dtype* data) {
  CUDA_KERNEL_LOOP(index, num * dim) {
    int n = index / dim;
    data[index] /= scale[n];
  }
}


template <typename Dtype>
Dtype DotProductSimilarityLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  const Dtype* weight = this->simvec_.gpu_data();
  // kernel_get_max<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
  //    num, dim, bottom_data, scale_data);
  // do inner product
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  // uniform vectors
  // sum
  caffe_gpu_gemv<Dtype>(CblasNoTrans, M_, K_, 1., bottom_data,
      sum_multiplier_.gpu_data(), 0., scale_data);
  // Do division
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_dotp_div<Dtype><<<CAFFE_GET_BLOCKS(M_ * N_),
                              CAFFE_CUDA_NUM_THREADS>>>(
      M_, N_, scale_data, top_data);
  return Dtype(0);
}

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) return;
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* scale_data = scale_.gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
      top_diff, this->simvec_.gpu_data(), (Dtype)0., bottom_diff);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  // cuda dot returns the result to cpu, so we temporarily change the pointer
  // mode
  CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
      CUBLAS_POINTER_MODE_DEVICE));
  Dtype* sub_data = sub_.mutable_gpu_data();
  for (int i = 0; i < M_; ++i) {
    caffe_gpu_dot<Dtype>(N_, top_diff + i * N_,
        top_data + i * N_, sub_data + i);
  }
  CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
      CUBLAS_POINTER_MODE_HOST));
  // subtraction
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, -1.,
      sub_.gpu_data(), sum_multiplier_.gpu_data(), 1., bottom_diff);
  // Do division
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_dotp_div<Dtype><<<CAFFE_GET_BLOCKS(M_ * K_),
                              CAFFE_CUDA_NUM_THREADS>>>(
      M_, K_, scale_data, bottom_diff);
}

INSTANTIATE_CLASS(DotProductSimilarityLayer);


}  // namespace caffe
