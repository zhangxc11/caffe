// Copyright 2014 BVLC and contributors.
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1)
    << "Dot-Product Similarity Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1)
    << "Dot-Product Similarity Layer takes a single blob as output.";

  BlobProto blob_proto;
  ReadProtoFromBinaryFile(
    this->layer_param_.dot_product_similarity_param().source(),
    &blob_proto);
  simvec_.FromProto(blob_proto);
  // Figure out the dimensions
  const int num_output = simvec_.height();
  M_ = bottom[0]->num();
  K_ = bottom[0]->count() / bottom[0]->num();
  N_ = num_output;
  CHECK_EQ(simvec_.num(), 1);
  CHECK_EQ(simvec_.channels(), 1);
  CHECK_EQ(simvec_.width(), K_)
    << "Dot-Product Similarity vector must have same dim with input.";

  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);

  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = Dtype(1.);
  }
  scale_.Reshape(bottom[0]->num(), 1, 1, 1);
  sub_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
Dtype DotProductSimilarityLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  const Dtype* weight = this->simvec_.cpu_data();
  // do inner product
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  // // uniform vectors
  // // sum
  // caffe_cpu_gemv<Dtype>(CblasNoTrans, M_, K_, 1., bottom_data,
  //     sum_multiplier_.cpu_data(), 0., scale_data);
  // // Do division
  // for (int i = 0; i < M_; ++i) {
  //   caffe_scal<Dtype>(N_, Dtype(1.) / scale_data[i], top_data + i * N_);
  // }
  return Dtype(0);
}

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) return;
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  Dtype* sub_data = sub_.mutable_cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
      top_diff, this->simvec_.cpu_data(), (Dtype)0., bottom_diff);
  // // Compute inner1d(top_diff, top_data)
  // // and subtract them from the bottom diff
  // for (int i = 0; i < M_; ++i) {
  //   sub_data[i] = caffe_cpu_dot<Dtype>(N_, top_diff + i * N_,
  //       top_data + i * N_);
  // }
  // // subtraction
  // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, -1.,
  //     sub_data, sum_multiplier_.cpu_data(), 1., bottom_diff);
  // // Do division
  // for (int i = 0; i < M_; ++i) {
  //   caffe_scal<Dtype>(K_, Dtype(1.) / scale_data[i], bottom_diff + i * K_);
  // }
}


INSTANTIATE_CLASS(DotProductSimilarityLayer);


}  // namespace caffe
