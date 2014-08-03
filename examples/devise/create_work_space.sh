#!/usr/bin/env sh
#

mkdir $1
cd $1

DATA=../../../data/devise

cp -r ../ilsvrc12_train_leveldb ilsvrc12_train_leveldb
cp -r ../ilsvrc12_val_leveldb ilsvrc12_val_leveldb
cp ../train_devise.sh .
cp ../resume_training.sh .
cp ../devise_solver.prototxt .
cp ../devise_train.prototxt .
cp ../devise_val.prototxt .
cp ../devise_solver_r.prototxt .
cp ../devise_train_r.prototxt .
ln -s ../wordvector.binaryproto wordvector1k.binaryproto
ln -s ../init.solverstate init.solverstate
ln -s ../../../data/ilsvrc12/imagenet_mean.binaryproto imagenet_mean.binaryproto
ln -s $DATA/caffe_reference_imagenet_model .

