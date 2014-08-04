#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Last modified: 2014 8月 04 13时12分42秒

"""docstring
"""

__revision__ = '0.1'

import numpy as np
import sys
from pylab import *

def main(argv):
    with open(argv[1]) as f:
        a = [l.strip().split(' ') for l in f.readlines()]
    x = [eval(l[5][:-1]) for l in a if l[3] == 'solver.cpp:87]']
    y = [eval(l[8]) for l in a if l[3] == 'solver.cpp:87]']
    tx = [eval(l[5][:-1]) for l in a if l[3] == 'solver.cpp:106]']
    ty = [eval(l[7]) for l in a if l[3] == 'solver.cpp:142]' and l[6] == '#0:']

    subplot(2, 1, 1)
    plot(np.array(x), np.array(y))
    subplot(2, 1, 2)
    plot(np.array(tx), np.array(ty))
    savefig('train.eps')

if __name__ == '__main__':
    main(sys.argv)
