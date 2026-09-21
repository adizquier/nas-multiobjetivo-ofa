from ofa2.tutorial.multi_objective_optimization import individual_to_arch, OfaIndividual, OfaRandomSampler
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice
import numpy as np

class OfaProblem(Problem):
    def __init__(self, efficiency_predictor, accuracy_predictor, search_vars):
        self.ks = Choice(options=search_vars.get('ks'))
        self.e = Choice(options=search_vars.get('e'))
        self.d = Choice(options=search_vars.get('d'))
        self.r = Choice(options=search_vars.get('r'))

        self.blocks = 20
        self.stages = 5

        super().__init__(
            vars={
                **{f"ks_{i}": self.ks for i in range(self.blocks)},
                **{f"e_{i}": self.e for i in range(self.blocks)},
                **{f"d_{i}": self.d for i in range(self.stages)},
                "r": self.r,
            },
            n_obj=2,
            n_constr=0,
        )
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.search_vars = search_vars

    def _evaluate(self, x, out):
        # x.shape = (population_size, n_var) = (100, 4)
        arch = individual_to_arch(x, self.blocks)

        f1 = self.efficiency_predictor.predict_efficiency(arch)

        f2 = 100 - self.accuracy_predictor.predict_accuracy(arch)
        out["F"] = np.column_stack([f1, f2])
    
class OfaSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return [
            [np.random.choice(var.options) for var in problem.vars.values()]
            for _ in range(n_samples)
        ]

class OfaSamplingPSO(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # Generamos enteros aleatorios respetando los límites xl y xu del problema.
        # size=(n_samples, n_var) devuelve la matriz correcta que espera Pymoo.
        # randint es exclusivo en el límite superior, por eso sumamos 1 a xu.
        return np.random.randint(problem.xl, problem.xu + 1, size=(n_samples, problem.n_var))

class OfaProblemPSO(Problem):

    def __init__(self, efficiency_predictor, accuracy_predictor, search_vars):
        self.ks_options = search_vars["ks"]
        self.e_options  = search_vars["e"]
        self.d_options  = search_vars["d"]
        self.r_options  = search_vars["r"]

        self.blocks = 20
        self.stages = 5

        n_var = self.blocks*2 + self.stages + 1

        # límites
        xl = []
        xu = []

        # ks
        for _ in range(self.blocks):
            xl.append(0)
            xu.append(len(self.ks_options)-1)

        # e
        for _ in range(self.blocks):
            xl.append(0)
            xu.append(len(self.e_options)-1)

        # d
        for _ in range(self.stages):
            xl.append(0)
            xu.append(len(self.d_options)-1)

        # r
        xl.append(0)
        xu.append(len(self.r_options)-1)

        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

    def _decode_individual(self, x):
        b = self.blocks
        s = self.stages

        ks = [ self.ks_options[int(i)] for i in x[0:b] ]
        e  = [ self.e_options[int(i)] for i in x[b:2*b] ]
        d  = [ self.d_options[int(i)] for i in x[2*b:2*b+s] ]

        r_value = int(self.r_options[int(x[-1])])
        r = [r_value]

        return {"ks": ks, "e": e, "d": d, "r": r}

    def _evaluate(self, X, out, *args, **kwargs):
        archs = [ self._decode_individual(ind) for ind in X ]

        f1 = self.efficiency_predictor.predict_efficiency(archs)

        f2 = 100 - self.accuracy_predictor.predict_accuracy(archs)

        out["F"] = np.column_stack([f1, f2])



