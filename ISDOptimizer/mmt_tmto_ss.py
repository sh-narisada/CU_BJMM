# MMT algorithm + TMTO + Schroeppel-Shamir technique (fixed version)

import collections
from basic_functionalities import *

set_vars = collections.namedtuple('MMT_tmto_ss', 'p l') # r is fix
num_vars=2
def inject(f) : return wrap(f, set_vars)

k = lambda x : 0.1
w_=Hi(1-k([0]))
w = lambda x : w_

r1 = lambda x: binomH(x.p,x.p/2)
L0 = lambda x: binomH((k(x)+x.l)/4,x.p/8)
r = lambda x: L0(x)
l1 = lambda x: 2*r(x)
L1 = lambda x: 2*L0(x) - r(x)
L2 = lambda x: 2*L1(x) - (l1(x) - r(x))

perms = lambda x: binomH(1., w(x)) - binomH(k(x)+x.l, x.p) - binomH(1-k(x)-x.l, w(x)-x.p)

constraints = [
{'type' : 'ineq', 'fun' : inject(lambda x : l1(x) - r(x))},
{'type' : 'ineq', 'fun' : inject(lambda x : x.l -l1(x))},
{'type' : 'ineq', 'fun' : inject(lambda x : 2*r(x) +l1(x) - r1(x))},
{'type' : 'ineq', 'fun' : inject(lambda x : (1. - k(x)- x.l) - (w(x) - x.p))},
{'type' : 'ineq', 'fun' : inject(lambda x : w(x) - x.p)},
]

def memory(x):
    x = set_vars(*x)
    return max(L0(x), L1(x),L2(x))

def time(x):
    x = set_vars(*x)
    time1= max(2*r(x) + l1(x)- r1(x),0)+max(L0(x), L1(x),L2(x),2*L2(x)-(x.l-l1(x)))
    return perms(x) + time1