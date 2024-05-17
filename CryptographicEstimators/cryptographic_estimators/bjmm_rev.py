# this file was newly added by the authors

from ...base_algorithm import optimal_parameter
from ..sd_algorithm import SDAlgorithm
from ..sd_problem import SDProblem
from ..sd_helper import _gaussian_elimination_complexity, _mem_matrix, _list_merge_complexity, min_max, \
    binom, log, log2, ceil, inf, _list_merge_async_complexity
from types import SimpleNamespace
from ..sd_constants import *
from enum import Flag, auto

class IMPL(Flag):
    M4RI = auto()
    MULT_W_INIT = auto()
    PARITY_BIT_TRICK = auto()

impl = IMPL.M4RI | IMPL.MULT_W_INIT

def lgC(a, b):
    if (a == 0 or b == 0 or a < 0 or b < 0 or a < b):
        return 0
    return log2(binom(a,b))

def lgSum(x, y):
    if x > y:
        return x + log2(1 + 2.0 ** (y - x))
    else:
        return y + log2(1 + 2.0 ** (x - y))
    

class BJMMrev(SDAlgorithm):
    def __init__(self, problem: SDProblem, **kwargs):
        super(BJMMrev, self).__init__(problem, **kwargs)
        self._name = "BJMMrev"
        self.initialize_parameter_ranges()
        self.limit_depth = kwargs.get("limit_depth", False)
        self.qc = kwargs.get("qc", 0)


    def initialize_parameter_ranges(self):
        """
        initialize the parameter ranges for p, l, l1 to start the optimisation
        process.
        """
        n, k, w = self.problem.get_parameters()
        s = self.full_domain
        self.set_parameter_ranges("p", 0, min_max(10, w, s))
        self.set_parameter_ranges("l", 0, min_max(500, n - k, s))
        self.set_parameter_ranges("l1", 0, min_max(200, n - k, s))

    @optimal_parameter
    def l(self):
        """
        Return the optimal parameter $l$ used in the algorithm optimization

        EXAMPLES::

            sage: from cryptographic_estimators.SDEstimator.SDAlgorithms import BJMMrev
            sage: from cryptographic_estimators.SDEstimator import SDProblem
            sage: A = BJMMrev(SDProblem(n=100,k=50,w=10))
            sage: A.l()
            8

        """
        return self._get_optimal_parameter("l")

    @optimal_parameter
    def l1(self):
        """
        Return the optimal parameter $l$ used in the algorithm optimization

        EXAMPLES::

            sage: from cryptographic_estimators.SDEstimator.SDAlgorithms import BJMMrev
            sage: from cryptographic_estimators.SDEstimator import SDProblem
            sage: A = BJMMrev(SDProblem(n=100,k=50,w=10))
            sage: A.l1()
            2

        """
        return self._get_optimal_parameter("l1")

    @optimal_parameter
    def p(self):
        """
        Return the optimal parameter $p$ used in the algorithm optimization

        EXAMPLES::

            sage: from cryptographic_estimators.SDEstimator.SDAlgorithms import BJMMrev
            sage: from cryptographic_estimators.SDEstimator import SDProblem
            sage: A = BJMMrev(SDProblem(n=100,k=50,w=10))
            sage: A.p()
            2
        """
        return self._get_optimal_parameter("p")

    def _are_parameters_invalid(self, parameters: dict):
        """
        return if the parameter set `parameters` is invalid

        """
        n, k, w = self.problem.get_parameters()
        par = SimpleNamespace(**parameters)
        k1 = (k + par.l) // 2
        if par.p > w // 2 or \
            k1 < par.p or \
            par.l >= n - k or\
            n - k - par.l < w - 2 * par.p or \
            k1 - par.p < 0 or \
            par.l1 > par.l:
            return True
        return False
 
    def _valid_choices(self):
        """
        Generator which yields on each call a new set of valid parameters based on the `_parameter_ranges` and already
        set parameters in `_optimal_parameters`
        """
        new_ranges = self._fix_ranges_for_already_set_parameters()

        n, k, w = self.problem.get_parameters()

        for p in range(new_ranges["p"]["min"], min(w // 2, new_ranges["p"]["max"]), 2):
            for l in range(new_ranges["l"]["min"], min(n - k - (w - 2 * p), new_ranges["l"]["max"])):
                L1 = log2(binom((k+l)//2, p // 2))
                d1 = self._adjust_radius
                for l1 in range(max(int(L1)-d1, 0), int(L1)+d1):
                    indices = {"p": p, "l": l, "l1": l1,
                                "r": self._optimal_parameters["r"]}
                    if self._are_parameters_invalid(indices):
                        continue
                    yield indices

    def _time_and_memory_complexity(self, parameters: dict, verbose_information=None):
        """
        computes the expected runtime and memory consumption for the depth 2 version

        """
        n, k, w = self.problem.get_parameters()
        par = SimpleNamespace(**parameters)
        k1 = (k + par.l) // 2

        if self._are_parameters_invalid(parameters):
            return inf, inf

        solutions = self.problem.nsolutions

        L1 = 0
        if impl & IMPL.MULT_W_INIT:
            for i in range(0, par.p // 2 + 1):
                L1 += binom(k1, i)
        else:
            L1 = binom(k1, par.p // 2)

        if self.qc:
            L1b = 0
            if impl & IMPL.MULT_W_INIT:
                for i in range(0, par.p // 2):
                    L1b += k * binom(k1, i)
            else:
                L1b = k * binom(k1, par.p // 2 - 1)
  
        if self._is_early_abort_possible(log2(L1)):
            return inf, inf

        L12 = max(1, L1 ** 2 // 2 ** par.l1)
        qc_advantage = 0
        if self.qc:
            L12b = max(1, L1 * L1b // 2 ** par.l1)
            qc_advantage = log2(k)

        memory = log2((2 * L1 + L12) + _mem_matrix(n, k, par.r)) if not self.qc else\
                  log2(L1 + L1b + min(L12, L12b) + _mem_matrix(n, k, par.r))
        if self._is_early_abort_possible(memory):
            return inf, inf

        Tg = _gaussian_elimination_complexity(n, k, par.r)

        if not self.qc:
            T_tree = 2 * _list_merge_complexity(L1, par.l1, self._hmap) +\
                         _list_merge_complexity(L12, par.l - par.l1, self._hmap)
        else:
            T_tree = _list_merge_async_complexity(L1, L1b, par.l1, self._hmap) +\
                     _list_merge_complexity(L1, par.l1, self._hmap) +\
                     _list_merge_async_complexity(L12, L12b, par.l - par.l1, self._hmap)

        R = [0]* (par.p + 1)
        for i in range(0, par.p + 1, 1):
            R[i] = binom(i, i // 2) * binom(k1 - i, par.p / 2 - ceil(i / 2))
            if impl & IMPL.MULT_W_INIT:
                for p1 in range(0, par.p//2 + 1, 1):
                    for p2 in range(0, par.p//2 + 1, 1):
                        if p1 == par.p//2 and p2== par.p//2:
                            continue
                        if abs(p1 - p2) <= i and i <= p1 + p2:
                            R[i] += binom(i, i // 2) * binom(k1 - i, (p1 + p2 - i)//2)
        if self.qc:
            Rb = [0]* (par.p)
            for i in range(0, par.p, 1):
                Rb[i] = binom(i, i // 2) * binom(k1 - i, (par.p - 1 - i)//2)
                if impl & IMPL.MULT_W_INIT:
                    for p1 in range(0, par.p//2 + 1, 1):
                        for p2 in range(0, par.p//2, 1):
                            if p1 == par.p//2 and p2== par.p//2 - 1:
                                continue
                            if abs(p1 - p2) <= i and i <= p1 + p2:
                                Rb[i] += binom(i, i // 2) * binom(k1 - i, (p1 + p2 - i)//2)

        p_step = 1 if impl & IMPL.MULT_W_INIT else 2
        cand_exp = 0
        if not self.qc:
            for i in range(0, par.p + 1, p_step):
                for j in range(i, par.p + 1, p_step):
                    if i == 0 and j == 0:
                        p_succ = min(0, - 2 * par.l1 + log2(R[i]) + log2(R[j]))
                    else:
                        p_succ = min(0, - par.l1 + log2(R[i]) + log2(R[j]))

                    cand_ij = lgC(k1, i) + lgC(k1, j) + lgC(n - k - par.l, w - i - j)
                    cand_ij_exp = max(0, cand_ij + p_succ)
                    cand_exp = lgSum(cand_exp, cand_ij_exp)
                    if i != j:
                        cand_exp = lgSum(cand_exp, cand_ij_exp)
        else:
            for i in range(0, par.p + 1, p_step):
                for j in range(0, par.p, p_step):
                    if i == 0 and j == 0:
                        p_succ = min(0, - 2 * par.l1 + log2(R[i]) + log2(Rb[j]))
                    else:
                        p_succ = min(0, - par.l1 + log2(R[i]) + log2(Rb[j]))
                    cand_ij = lgC(k1, i) + lgC(k1, j) + lgC(n - k - par.l, w - i - j) + qc_advantage
                    cand_ij_exp = max(0, cand_ij + p_succ)
                    cand_exp = lgSum(cand_exp, cand_ij_exp)

        Tp = max(log2(binom(n, w)) - cand_exp - solutions, 0)
        N_HAT = Tp
        time = N_HAT + log2(Tg + T_tree)

        if verbose_information is not None:
            verbose_information[VerboseInformation.CONSTRAINTS.value] = [par.l1, par.l - par.l1]
            verbose_information[VerboseInformation.PERMUTATIONS.value] = N_HAT
            verbose_information[VerboseInformation.TREE.value] = log2(T_tree)
            verbose_information[VerboseInformation.GAUSS.value] = log2(Tg)
            verbose_information[VerboseInformation.LISTS.value] = [
                log2(L1), log2(L12), 2 * log2(L12) - (par.l - par.l1)]

        return time, memory

    def __repr__(self):
        rep = "BJMM+ estimator for " + str(self.problem)
        return rep
