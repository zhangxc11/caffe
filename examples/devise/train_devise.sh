#!/usr/bin/env sh

TOOLS=$CAFFE_ROOT/build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    devise_solver.prototxt init.solverstate

echo "Done."
