#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    devise_solver_r.prototxt caffe_devise_train_iter_20000.solverstate

echo "Done."
