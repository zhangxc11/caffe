#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Last modified: 2014 8月 05 16时25分24秒

"""docstring
"""

__revision__ = '0.1'

import numpy as np
import sys
import matplotlib.pyplot as plt

def main(argv):
    with open(argv[1]) as f:
        a = [l.strip().split(' ') for l in f.readlines()]

    x = [eval(l[5][:-1]) for l in a if len(l) > 7 and l[6] == 'loss' and l[7] == '=']
    y = [eval(l[8]) for l in a if len(l) > 8 and l[6] == 'loss' and l[7] == '=']
    lr = [eval(l[8]) for l in a if len(l) > 8 and l[6] == 'lr' and l[7] == '=']
    tx = [eval(l[5][:-1]) for l in a if len(l) > 6 and l[6] == 'Testing']
    ty = [eval(l[7]) for l in a if len(l) > 7 and l[6] == '#0:']

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()
    ax1.plot(np.array(x), np.array(y))
    ax1.set_ylabel('training loss')
    ax2.plot(np.array(x), np.array(lr), 'g')
    ax2.set_ylabel('learning rate')
    # plotyy(np.array(x), np.array(y), np.array(x), np.array(lr))
    print 'train log #', len(x)
    if len(argv) > 3:
        ax1.set_title(argv[3])
    ax3 = fig.add_subplot(212)
    ax3.plot(np.array(tx), np.array(ty))
    ax3.set_ylabel('test rate')
    print 'val log #', len(tx)
    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.savefig('train.eps')

if __name__ == '__main__':
    main(sys.argv)
