#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Last modified: 2014 8月 04 21时11分05秒

"""docstring
"""

__revision__ = '0.1'

import numpy as np
import sys
from pylab import *

def main(argv):
    with open(argv[1]) as f:
        a = [l.strip().split(' ') for l in f.readlines()]

    x = [eval(l[5][:-1]) for l in a if len(l) > 7 and l[6] == 'loss' and l[7] == '=']
    y = [eval(l[8]) for l in a if len(l) > 8 and l[6] == 'loss'and l[7] == '=']
    tx = [eval(l[5][:-1]) for l in a if len(l) > 6 and l[6] == 'Testing']
    ty = [eval(l[7]) for l in a if len(l) > 7 and l[6] == '#0:']

    subplot(2, 1, 1)
    plot(np.array(x), np.array(y))
    print 'train log #', len(x)
    subplot(2, 1, 2)
    plot(np.array(tx), np.array(ty))
    print 'val log #', len(tx)
    if len(argv) > 2:
        savefig(argv[2])
    else:
        savefig('train.eps')

if __name__ == '__main__':
    main(sys.argv)
