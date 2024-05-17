# Dumer's algorithm + Schroeppel-Shamir technique

import collections
from basic_functionalities import *

set_vars = collections.namedtuple('Dumer_ss', 'p l')
num_vars=2
def inject(f) : return wrap(f, set_vars)

k = lambda x : 0.1
w_=Hi(1-k([0]))
w = lambda x : w_

L0 = lambda x: binomH((k(x)+x.l)/4,x.p/4)
r = lambda x: binomH((k(x)+x.l)/4,x.p/4)
L1 = lambda x: 2*L0(x) - r(x)

perms = lambda x: binomH(1., w(x)) - binomH(k(x)+x.l, x.p) - binomH(1-k(x)-x.l, w(x)-x.p)

constraints = [
{'type' : 'ineq', 'fun' : inject(lambda x : x.l - r(x))},
{'type' : 'ineq', 'fun' : inject(lambda x : (1. - k(x)- x.l) - (w(x) - x.p))},
{'type' : 'ineq', 'fun' : inject(lambda x : w(x) - x.p)},
]

def memory(x):
    x = set_vars(*x)
    return max(L0(x),L1(x))

def time(x):
    x = set_vars(*x)
    time1=r(x)+max(L0(x),L1(x),2*L1(x)-(x.l- r(x)))
    return perms(x) + time1