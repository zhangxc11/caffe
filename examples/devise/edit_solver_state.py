#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Last modified: 2014 8月 05 09时18分40秒

"""docstring
"""

__revision__ = '0.1'
import caffe
import sys


def main(argv):
    st = caffe.proto.caffe_pb2.SolverState()
    with open(argv[1], 'rb') as f:
        st.ParseFromString(f.read())
    st.iter = 0
    for d in st.history:
        for i in range(d.num * d.channels * d.height * d.width):
            d.data[i] = 0.
    if len(argv > 2):
        st.learned_net = argv[2]
    if len(argv > 3):
        out = argv[3]
    else:
        out = argv[1]
    with open(out, 'wb') as f:
        f.write(st.SerializeToString())

if __name__ == '__main__':
    main(sys.argv)
