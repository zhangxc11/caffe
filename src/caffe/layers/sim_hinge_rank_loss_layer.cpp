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
void SimHingeRankLossLayer<Dtype>::FurtherSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  BlobProto blob_proto;
  ReadProtoFromBinaryFile(
    this->layer_param_.sim_hinge_rank_loss_param().source(), &blob_proto);
  simvec_.FromProto(blob_proto);
  CHECK_EQ(simvec_.num(), 1);
  CHECK_EQ(simvec_.channels(), 1);
}


template <typename Dtype>
Dtype SimHingeRankLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
// TODO: to edit as SimHingeRankLossLayer
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* simvec_mat = simvec_.cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(simvec_.height(), dim);
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
      loss -= simvec_mat[label * dim + j] * log(prob);
    }
  }
  return loss / num;
}

template <typename Dtype>
void SimHingeRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
// TODO: to edit as SimHingeRankLossLayer~
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  const Dtype* simvec_mat = simvec_.cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  CHECK_EQ(simvec_.height(), dim);
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
      bottom_diff[i * dim + j] = - simvec_mat[label * dim + j] / prob / num;
    }
  }
}

INSTANTIATE_CLASS(SimHingeRankLossLayer);

}  // namespace caffe
