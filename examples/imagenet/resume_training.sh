#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    imagenet_solver_r.prototxt caffe_imagenet_train_90000.solverstate

echo "Done."
