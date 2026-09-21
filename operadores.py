'''
Archivo que implementa los operadores proprios de muestreo, cruce y mutación tras el análisis OFAT.

El análisis OFAT muestra que los valores bajos de cada hiperparámetro maximizan la eficiencia accuracy/latencia.
Los operadores diseñados sesgan la búsqueda hacia valores pequeños.

MUESTREO: cada gen se inicializa con un valor aleatorio de su dominio, garantizando arquitecturas factibles.

CRUCE: dados dos padres, el hijo hereda en cada posición el menor de los dos valores.

MUTACIÓN: se muta un solo gen, el más cercano a su máximo (el más caro en latencia) que aún pueda bajar, reasignándolo a un valor inferior.
'''

import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation


class OfaSampling(Sampling):
    '''
    Inicializa cada gen con un valor aleatorio de su propio dominio discreto, garantizando que todo individuo sea factible
    '''

    def _do(self, problem, n_samples, **kwargs):
        '''
        Genera la población inicial que utiliza pymoo.

        Args:
            problem (OfaProblem): problema con las variables del espacio de búsqueda
            n_samples (int): numero de individuos a generar

        Returns:
            list: n_samples individuos de 46 genes
        '''

        return [
            [np.random.choice(var.options) for var in problem.vars.values()]
            for _ in range(n_samples)
        ]


class OfaMinCrossover(Crossover):
    '''
    El hijo hereda en cada posición el menor de los dos valores parentales, sesgando la
    descendencia hacia arquitecturas más eficientes.
    '''

    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=1, **kwargs)

    def _do(self, _, X, random_state=None, **kwargs):
        '''
        Genera un hijo por pareja de progenitores.

        Args:
            X (ndarray): progenitores, con los dos padres en las dos primeras posiciones

        Returns:
            ndarray: descendencia con la dimensión de offspring añadida
        '''
        p1, p2 = X[0], X[1]
        child = np.where(p1 <= p2, p1, p2)
        return child[None, ...]


class OfaBiasedCrossover(Crossover):
    '''
    Variante probabilística del cruce: en cada posición elige el menor de los dos valores con probabilidad p_min (1 - p_min el mayor).
    Mantiene el sesgo hacia valores bajos pero conserva diversidad y, por tanto, cobertura del frente.

    p_min = 1.0 equivale a OfaMinCrossover
    '''

    def __init__(self, p_min=0.7, **kwargs):
        super().__init__(n_parents=2, n_offsprings=1, **kwargs)
        self.p_min = p_min

    def _do(self, _, X, random_state=None, **kwargs):
        '''
        Genera la descendencia de todos los emparejamientos de la generación, decidiendo gen a gen si se hereda el menor o el mayor 
        de los dos valores parentales.

        Args:
            X (ndarray): progenitores, con forma (2, n_matings, 46)
            random_state: generador aleatorio que inyecta pymoo

        Returns:
            ndarray: descendencia con forma (1, n_matings, 46)
        '''
        p1, p2 = X[0], X[1]
        lo = np.where(p1 <= p2, p1, p2)
        hi = np.where(p1 <= p2, p2, p1)
        take_lo = random_state.random(p1.shape) < self.p_min
        child = np.where(take_lo, lo, hi)
        return child[None, ...]


class OfaGuidedMutation(Mutation):
    '''
    Muta un único gen por individuo, siendo este el más cercano a su máximo  que aún pueda bajar, reasignándolo a un valor inferior.
    '''

    def _do(self, problem, X, random_state=None, **kwargs):
        '''
        Aplica la mutación a cada individuo de la población.

        Args:
            problem (OfaProblem): problema con las variables del espacio de búsqueda
            X (ndarray): población con forma (n_individuos, 46)
            random_state: generador aleatorio que inyecta pymoo

        Returns:
            ndarray: población mutada
        '''
        
        assert problem.vars is not None
        X = X.astype(object)

        sorted_opts = [np.sort(np.asarray(var.options)) for var in problem.vars.values()]

        for i in range(len(X)):
            positions = np.array([
                (np.searchsorted(opts, X[i, k]) / (len(opts) - 1))
                for k, opts in enumerate(sorted_opts)
            ])

            reducible = np.where(positions > 0)[0]
            if len(reducible) == 0:
                continue

            k = reducible[np.argmax(positions[reducible])]
            opts = sorted_opts[k]
            lower = opts[:np.searchsorted(opts, X[i, k])]
            X[i, k] = random_state.choice(lower)

        return X