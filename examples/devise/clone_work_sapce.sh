#!/usr/bin/env sh
#

mkdir $2

cp $1/train_devise.sh $2
cp $1/resume_training.sh $2
cp $1/devise_solver.prototxt $2
cp $1/devise_train.prototxt $2
cp $1/devise_val.prototxt $2
cp $1/devise_solver_r.prototxt $2
cp $1/devise_train_r.prototxt $2
cp -d $1/init.solverstate $2
cp -d $1/caffe_reference_imagenet_model $2
cp -d $1/imagenet_mean.binaryproto $2
cp -d $1/wordvector1k.binaryproto $2
cp -r $1/ilsvrc12_train_leveldb $2
cp -r $1/ilsvrc12_val_leveldb $2
