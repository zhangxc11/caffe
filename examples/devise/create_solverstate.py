#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Last modified: 2014 8月 03 12时02分01秒

"""docstring
"""

__revision__ = '0.1'

import caffe

model = '../../data/devise/caffe_reference_imagenet_model';
state = 'init.solverstate'
net = caffe.proto.caffe_pb2.NetParameter()
f = open(model, 'rb')
net.ParseFromString(f.read())
f.close()
st = caffe.proto.caffe_pb2.SolverState()
st.iter = 0
st.learned_net = model

for layer in net.layers:
    for blob in layer.blobs:
        bl = st.history.add()
        bl.num = blob.num
        bl.channels = blob.channels
        bl.height = blob.height
        bl.width = blob.width
        bl.data.extend(blob.data)
        for i in range(len(bl.data)):
            bl.data[i] = 0

f = open(state, 'wb')
f.write(st.SerializeToString())
f.close()

