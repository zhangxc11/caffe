#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Last modified: 2014 8月 05 16时26分48秒

"""docstring
"""

__revision__ = '0.1'

import struct
import caffe
import numpy.random as rd
from math import sqrt

wordlist = '../../data/devise/wordlist.txt'
wordvec = '../../data/devise/wordvec.bin'
imgwords = '../../data/ilsvrc12/synset_words.txt'

with open(wordlist) as f:
    words, dim = [eval(i) for i in f.readline().strip().split(' ')]
    wordlist = [w.strip() for w in f.readlines()]
    print 'words #', words, 'dim:', dim

with open(imgwords) as f:
    name = ['_'.join(l.strip().split(' ')[1:]).split(',') for l in f.readlines()]

with open(wordvec, 'rb') as f:
    vector = [struct.unpack('f'*dim, f.read(4 * dim)) for i in range(words)]

blob = caffe.proto.caffe_pb2.BlobProto()
blob.num = 1
blob.channels = 1
blob.width = dim
blob.height = len(name)

ind = [-1 for i in range(len(name))]
err = 0
for i in range(len(name)):
    for n in name[i]:
        n = n.lower()
        if n.startswith('_'):
            n = n[1:]
        if n in wordlist:
            ind[i] = wordlist.index(n)
            for v in vector[ind[i]]:
                blob.data.append(v)
            break
    if ind[i] < 0:
        vec = []
        sum = 0
        for v in range(dim):
            a = rd.rand() - 0.5
            sum = sum + a * a
            vec.append(a)
        sum = sqrt(sum)
        for v in range(dim):
            vec[v] = vec[v] / sum
            blob.data.append(vec[v])
        err = err + 1

print err
#for i in range(1000000):
#    blob.data.append(rd.rand())

with open('wordvector.binaryproto', 'wb') as f:
    f.write(blob.SerializeToString())
