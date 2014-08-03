#!/usr/bin/env sh
#

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

scp xingcheng@jouer.smile.usc.edu:~/Code/caffe/data/devise/wordlist.txt .
scp xingcheng@jouer.smile.usc.edu:~/Code/caffe/data/devise/wordvec.bin .
scp xingcheng@jouer.smile.usc.edu:~/Code/caffe/data/devise/caffe_reference_imagenet_model .


echo "Done."
