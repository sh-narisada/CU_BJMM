# depth-2 BJMM algorithm + TMTO = Revisited MMT algorithm

import collections
from basic_functionalities import *

set_vars = collections.namedtuple('BJMM_d2_tmto', 'p d1 l')
num_vars=3
def inject(f) : return wrap(f, set_vars)

k = lambda x : 0.1
w_=Hi(1-k([0]))
w = lambda x : w_

p1 = lambda x: x.p/2 + x.d1
r1 = lambda x: reps(x.p, x.d1, k(x)+x.l)
l1 = lambda x: L1(x)
L1 = lambda x: binomH((k(x)+x.l)/2,p1(x)/2)
L2 = lambda x: 2*L1(x) -l1(x)

perms = lambda x: binomH(1., w(x)) - binomH(k(x)+x.l, x.p) - binomH(1-k(x)-x.l, w(x)-x.p)

constraints = [
{'type' : 'ineq', 'fun' : inject(lambda x : x.l - l1(x))},
{'type' : 'eq',   'fun' : inject(lambda x : r1(x) - l1(x))},
{'type' : 'ineq', 'fun' : inject(lambda x : (1. - k(x)- x.l) - (w(x) - x.p))},
{'type' : 'ineq', 'fun' : inject(lambda x : w(x) - x.p)},
]

def memory(x):
    x = set_vars(*x)
    return max(L1(x),L2(x))

def time(x):
    x = set_vars(*x)
    time1=max(l1(x)-r1(x),0)+max(L1(x),L2(x),2*L2(x)-(x.l-l1(x)))
    return perms(x) + time1