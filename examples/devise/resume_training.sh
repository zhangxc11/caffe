#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    imagenet_solver.prototxt caffe_imagenet_train_iter_60000.solverstate

echo "Done."
