// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/io.hpp"

using std::string;

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class DotProductSimilarityLayerTest : public ::testing::Test {
 protected:
  DotProductSimilarityLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()),
        blob_vec_(new Blob<Dtype>(1, 1, 20, 60)),
        blob_result_(new Blob<Dtype>(1, 1, 2, 20)),
        filename_(new string(tmpnam(NULL))) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_vec_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    createSourceFile();
    calulatResult();
  }
  void createSourceFile() {
    LOG(INFO) << "Using temporary source file " << *filename_;
    BlobProto blob_proto;
    blob_vec_->ToProto(&blob_proto);
    WriteProtoToBinaryFile(blob_proto, *filename_);
  }
  void calulatResult() {
    Dtype *result_data = blob_result_->mutable_cpu_data();
    const Dtype *bottom_data = blob_bottom_->cpu_data();
    const Dtype *vec_data = blob_vec_->cpu_data();
    const int M = 2;
    const int N = 20;
    const int K = 60;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        result_data[i * N + j] = 0;
        for (int k = 0; k < K; k++) {
          result_data[i * N + j] +=
            bottom_data[i * K + k] * vec_data[j * K + k];
        }
      }
    }
  }
  virtual ~DotProductSimilarityLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_vec_;
    delete blob_result_;
  }

  shared_ptr<string> filename_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_vec_;
  Blob<Dtype>* const blob_result_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(DotProductSimilarityLayerTest, Dtypes);

TYPED_TEST(DotProductSimilarityLayerTest, TestSetUp) {
  LayerParameter layer_param;
  DotProductSimilarityParameter* dot_product_similarity_param =
      layer_param.mutable_dot_product_similarity_param();
  dot_product_similarity_param->set_source(*(this->filename_));
  shared_ptr<DotProductSimilarityLayer<TypeParam> > layer(
      new DotProductSimilarityLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 20);
}

TYPED_TEST(DotProductSimilarityLayerTest, TestCPU) {
  LayerParameter layer_param;
  DotProductSimilarityParameter* dot_product_similarity_param =
      layer_param.mutable_dot_product_similarity_param();
  Caffe::set_mode(Caffe::CPU);
  dot_product_similarity_param->set_source(*(this->filename_));
  shared_ptr<DotProductSimilarityLayer<TypeParam> > layer(
      new DotProductSimilarityLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const TypeParam* result = this->blob_result_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_LE(abs(result[i] - data[i]), 1e-3);
  }
}

TYPED_TEST(DotProductSimilarityLayerTest, TestGPU) {
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    DotProductSimilarityParameter* dot_product_similarity_param =
      layer_param.mutable_dot_product_similarity_param();
    Caffe::set_mode(Caffe::GPU);
    dot_product_similarity_param->set_source(*(this->filename_));
    shared_ptr<DotProductSimilarityLayer<TypeParam> > layer(
        new DotProductSimilarityLayer<TypeParam>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
    layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
    const TypeParam* data = this->blob_top_->cpu_data();
    const TypeParam* result = this->blob_result_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_LE(abs(result[i] - data[i]), 1e-3);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(DotProductSimilarityLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  DotProductSimilarityParameter* dot_product_similarity_param =
      layer_param.mutable_dot_product_similarity_param();
  Caffe::set_mode(Caffe::CPU);
  dot_product_similarity_param->set_source(*(this->filename_));
  DotProductSimilarityLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(DotProductSimilarityLayerTest, TestGPUGradient) {
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    DotProductSimilarityParameter* dot_product_similarity_param =
        layer_param.mutable_dot_product_similarity_param();
    Caffe::set_mode(Caffe::GPU);
    dot_product_similarity_param->set_source(*(this->filename_));
    DotProductSimilarityLayer<TypeParam> layer(layer_param);
    GradientChecker<TypeParam> checker(1e-2, 1e-2);
    checker.CheckGradient(&layer, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_));
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
