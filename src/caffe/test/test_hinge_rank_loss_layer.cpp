// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class HingeRankLossLayerTest : public ::testing::Test {
 protected:
  HingeRankLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 20, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
  }
  virtual ~HingeRankLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(HingeRankLossLayerTest, Dtypes);


TYPED_TEST(HingeRankLossLayerTest, TestGPU) {
  int deviceNum = 0;
  cudaGetDeviceCount(&deviceNum);
  ASSERT_GT(deviceNum, 0)
      << "Failed due to no supported graphic card.";
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    Caffe::set_mode(Caffe::CPU);
    layer_param.mutable_hinge_rank_loss_param()->set_margin(1.);
    HingeRankLossLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    TypeParam loss = layer.Forward(this->blob_bottom_vec_,
        &this->blob_top_vec_);
    Caffe::set_mode(Caffe::GPU);
    HingeRankLossLayer<TypeParam> gpulayer(layer_param);
    gpulayer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    TypeParam gpuloss = gpulayer.Forward(this->blob_bottom_vec_,
        &this->blob_top_vec_);
    EXPECT_NEAR(loss, gpuloss, fabs(loss) * 1e-4);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(HingeRankLossLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  layer_param.mutable_hinge_rank_loss_param()->set_margin(1.);
  HingeRankLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 1, 0.01);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

TYPED_TEST(HingeRankLossLayerTest, TestGradientGPU) {
  int deviceNum = 0;
  cudaGetDeviceCount(&deviceNum);
  ASSERT_GT(deviceNum, 0)
      << "Failed due to no supported graphic card.";
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    Caffe::set_mode(Caffe::GPU);
    layer_param.mutable_hinge_rank_loss_param()->set_margin(1.);
    HingeRankLossLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 1, 0.01);
    checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_), 0, -1, -1);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
