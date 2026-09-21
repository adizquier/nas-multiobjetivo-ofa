"""
En este archivo se desarrolla la adaptación del algoritmo VNS al problema multiobjetivo tratado en este trabajo. Se utiliza como esquema 
algorítmico de referencia el algoritmo desarrollado en el artículo "A multi-objective metaheuristic approach for generating cell type marker 
panels from single-cell transcriptomics", que se basa en VNS.
"""

import numpy as np
import random
from pymoo.functions import FunctionLoader
import itertools


def cleanSolvedList(savedSols: list, tabu_set: list) -> list:
    '''
    Retorna una lista resultado de eliminar de savedSols los elementos de la lista tabú.

    NOTA: no se utiliza en la versión final del algoritmo. Corresponde a la
    primera iteración de diseño del algoritmo propuesto.

    Args:
        savedSols (list): lista de individuos
        tabuList (list): lista de individuos que se excluyen

    Returns:
        list: subconjunto de savedSols formado por los individuos que no
            están presentes en tabuList.

    '''
    return [sol for sol in savedSols if tuple(sol[0]) not in tabu_set]


def iniSol(search_space: dict) -> list:
    '''
    Inicialización de una solución (individuo con 46 genes) de manera aleatoria entre los posibles valores válidos.

    Args:
        searchSpace (dict): diccionario con los posibles valores validos para cada hp

    Returns:
        list: retorna una solucion
    '''
    ks = [random.choice(search_space['ks']) for _ in range(20)]
    e  = [random.choice(search_space['e'])  for _ in range(20)]
    d  = [random.choice(search_space['d'])  for _ in range(5)]
    r  = [random.choice(search_space['r'])]

    sol = ks + e + d + r
    return sol


def init_pob(search_space: dict, acc_predictor, latency_table):
    '''
    Genera la población inicial evaluando todas las combinaciones (ks, e, d, r) en las que cada hiperparámetro 
    toma un único valor constante en todas sus posiciones del vector.

    Args:
        search_space (dict): posibles valores validos para cada hp
        acc_predictor: predictor de accuracy de OFA
        latency_table: tabla de latencia del dispositivo

    Returns:
        tuple: (dict) población inicial, (int) evaluaciones consumidas
    '''

    ks_options = search_space.get("ks")
    e_options = search_space.get("e")
    d_options = search_space.get("d")
    rel_options = search_space.get("r")

    pob = {}
    n_eval = 0
    for ks, e, d, rel in itertools.product(ks_options, e_options, d_options, rel_options):
        indv = [ks]*20 + [e]*20 + [d]*5 + [rel]
        lat, acc = calculateAccLat(indv, acc_predictor, latency_table)

        sol = indv, lat, acc
        n_eval += 1

        try_add_pending(sol, pob)

    return pob, n_eval


def mutSol(sol: list, search_space: dict) -> list:
    '''
    Genera un vecino de sol modificando un único gen elegido al azar, al que se asigna otro valor válido de su hiperparámetro.

    Args:
        sol (list): individuo de 46 genes
        search_space (dict): posibles valores validos para cada hp

    Returns:
        list: nuevo individuo, copia de sol con un gen modificado
    '''
    new_sol = sol.copy()
    gen = np.random.randint(0, 46)

    if gen < 20:
        options = search_space['ks']
    elif gen < 40:
        options = search_space['e']
    elif gen < 45:
        options = search_space['d']
    else:
        options = search_space['r']

    new_sol[gen] = random.choice(options)
    return new_sol


def newMutSol(sol: list, search_space: dict, maxNeigh:int, neighCount: int = 1) -> list:
    '''
    Genera un vecino de sol mutando varios genes. El número de genes a mutar se calcula por interpolación lineal
    entre 1 (vecindad 1) y 23, la mitad del individuo.

    Args:
        sol (list): individuo de 46 genes
        search_space (dict): posibles valores validos para cada hp
        maxNeigh (int): numero de estructuras de vecindad
        neighCount (int): vecindad actual

    Returns:
        list: nuevo individuo con n_genes modificados
    '''
    new_sol = sol.copy()
    n_genes = round(1 + (neighCount - 1) * 22 / (maxNeigh - 1))
    posiciones = random.sample(range(46), n_genes)
    for gen in posiciones:
        if gen < 20:
            options = search_space['ks']
        elif gen < 40:
            options = search_space['e']
        elif gen < 45:
            options = search_space['d']
        else:
            options = search_space['r']
        new_sol[gen] = random.choice(options)
    return new_sol


def dominate(indv_a: tuple, indv_b: tuple) -> int:
    '''
    Comprueba la relación de dominancia entre dos individuos.

    Args:
        indv_a (tuple): individuo (genotipo, latencia, error)
        indv_b (tuple): individuo (genotipo, latencia, error)

    Returns:
        int:  1 si a domina a b
             -1 si b domina a a
              0 si ninguno domina al otro
    '''
    _, lat_a, err_a = indv_a
    _, lat_b, err_b = indv_b

    a_mejor = (lat_a < lat_b) or (err_a < err_b)
    b_mejor = (lat_b < lat_a) or (err_b < err_a)

    if a_mejor and not b_mejor:
        return 1
    if b_mejor and not a_mejor:
        return -1
    return 0


def decode(sol: list) -> dict:
    '''
    Traduce el genotipo plano de 46 genes a un dict por hiperparámetro, que es el formato que esperan los predictores de OFA.

    Args:
        sol (list): genotipo de 46 genes de un individuo.

    Returns:
        dict: arquitectura con claves "ks", "e", "d" y "r".
    '''
    return {
        "ks": sol[0:20],
        "e":  sol[20:40],
        "d":  sol[40:45],
        "r":  [sol[45]],
    }


def calculateAccLat(sol, acc_predictor, latency_table):
    '''
    Evalúa un individuo con los predictores y devuelve sus dos objetivos, ambos minimizados.

    Args:
        sol (list): genotipo de 46 genes de un individuo.
        acc_predictor: predictor de accuracy de OFA.
        latency_table: predictor de latencia para el hardware elegido.

    Returns:
        tuple: (lat, acc) donde
            lat (float): latencia estimada
            acc (float): error = 100 - accuracy
    '''
    arch = decode(sol)

    lat = float(latency_table.predict_efficiency([arch])[0])
    acc = 100 - acc_predictor.predict_accuracy([arch])[0].item()

    return lat, acc

def non_dominant_sorting(savedSols: list) -> list:
    '''
    Filtra savedSols devolviendo únicamente las soluciones no dominadas, es decir, el frente de Pareto del conjunto.

    Args:
        savedSols (list): lista de individuos evaluados

    Returns:
        list: subconjunto de savedSols formado por las soluciones no dominadas
    '''
    non_dominated_sols = []

    for sol in savedSols:
        dominated = False
        for sol_com in savedSols:
            if dominate(sol, sol_com) == -1:
                dominated = True
                break
        
        if not dominated:
            non_dominated_sols.append(sol)
    
    return non_dominated_sols


#----------------------- TRAIDO DE PYMOO - DOCUMENTACIÓN REALIZADA POR CLAUDE (REVISADA) --------------------------

def das_dennis(n_partitions, n_dim):
    '''
    Genera un conjunto de puntos de referencia distribuidos uniformemente sobre
    el símplex de dimensión n_dim (método de Das-Dennis). Cada punto es un vector
    de componentes no negativas que suman 1, y representa una dirección en el
    espacio de objetivos. Se usan como rejilla de nichos para repartir la búsqueda.

    Args:
        n_partitions (int): número de particiones por dimensión. Controla la
            finura de la rejilla (más particiones -> más puntos de referencia).
        n_dim (int): número de objetivos (dimensión del símplex).

    Returns:
        np.ndarray: matriz (n_puntos, n_dim) con los puntos de referencia.
    '''
    if n_partitions == 0:
        # Caso borde: un único punto en el centro del símplex
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)   # vector de trabajo, se rellena en la recursión
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    '''
    Función recursiva auxiliar de das_dennis.

    Args:
        ref_dirs (list): acumulador donde se van guardando los puntos generados.
        ref_dir (np.ndarray): vector de referencia parcial que se está construyendo.
        n_partitions (int): número total de particiones (constante en la recursión).
        beta (int): particiones que quedan por repartir en este punto de la recursión.
        depth (int): dimensión actual que se está rellenando.
    '''
    if depth == len(ref_dir) - 1:
        # Última dimensión: recibe todo lo que sobra, garantizando que el vector sume 1
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])   # [None, :] lo convierte en fila (1, n_dim)
    else:
        # Probar todas las asignaciones posibles (de 0 a beta) para esta dimensión
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            # np.copy: cada rama trabaja sobre su propia copia para no pisarse entre sí
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def load_function(func_name=None, _type="auto"):
    '''
    Carga una función interna de pymoo por su nombre a través del FunctionLoader.

    Args:
        func_name (str): nombre de la función a cargar.
        _type (str): modo de carga (por defecto "auto").

    Returns:
        callable: la función solicitada, lista para invocarse.
    '''
    return FunctionLoader.get_instance().load(func_name, mode=_type)


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=0.0):
    '''
    Asocia cada solución a su nicho (punto de referencia)

    Args:
        F (np.ndarray): objetivos de las soluciones, matriz (n_soluciones, n_obj).
        niches (np.ndarray): puntos de referencia (n_niches, n_obj), salida de das_dennis.
        ideal_point (np.ndarray): mejor valor por objetivo (mínimos).
        nadir_point (np.ndarray): peor valor por objetivo (máximos).
        utopian_epsilon (float): margen opcional para desplazar el punto ideal.

    Returns:
        tuple:
            niche_of_individuals (np.ndarray): índice del nicho asignado a cada solución.
            dist_to_niche (np.ndarray): distancia perpendicular de cada solución a su nicho.
            dist_matrix (np.ndarray): matriz completa de distancias (n_soluciones, n_niches).
    '''
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12   # evita división por cero si ideal y nadir coinciden en un objetivo

    # Normaliza los objetivos al rango definido por los puntos utópico y nadir
    N = (F - utopian_point) / denom
    dist_matrix = load_function("calc_perpendicular_distance")(N, niches)

    # Para cada solución, el nicho más cercano y su distancia a él
    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche, dist_matrix


def calc_niche_count(n_niches, niche_of_individuals):
    '''
    Cuenta cuántas soluciones han sido asignadas a cada nicho (punto de referencia).
    Sirve para identificar qué regiones del frente están más o menos pobladas.

    Args:
        n_niches (int): número total de nichos / puntos de referencia.
        niche_of_individuals (np.ndarray): nicho asignado a cada solución.

    Returns:
        np.ndarray: vector de longitud n_niches con el número de soluciones por nicho.
    '''
    niche_count = np.zeros(n_niches, dtype=int)
    # np.unique con return_counts da los nichos ocupados y cuántas soluciones tiene cada uno;
    # los nichos vacíos se quedan a 0 por la inicialización con zeros
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count

# --------------------------------------------------------------------------------------

def get_best_solution(P: list, ref_points) -> tuple:
    '''
    Selecciona una solución de P situada en la región menos poblada del frente, usando los puntos de referencia 
    de Das-Dennis como rejilla de nichos.

    Args:
        P (list): soluciones candidatas, cada una (genotipo, latencia, error).
        ref_pts (np.ndarray): puntos de referencia (n_niches, 2) de das_dennis.

    Returns:
        tuple: la solución elegida del nicho menos poblado.
    '''
    F = np.array([[sol[1], sol[2]] for sol in P])

    ideal = F.min(axis=0)
    nadir = F.max(axis=0)

    nichos, _, _ = associate_to_niches(F, ref_points, ideal, nadir)
    niche_count = calc_niche_count(len(ref_points), nichos)
    ocupados = np.unique(nichos)

    least_crowded = ocupados[np.argmin(niche_count[ocupados])]

    candidatos = np.where(nichos == least_crowded)[0]

    elegido = random.choice(candidatos)
    return P[elegido]


def try_add_pending(sol_new, conj: dict, tabuSet: set = None) -> None:
    '''
    Inserta sol_new en conj manteniendo solo el frente no dominado de las soluciones no exploradas. 
    No la inserta si está dominada por alguna ya presente o si está en tabu; elimina de conj las que sol_new domina.

    Args:
        sol_new (tuple): solución candidata (indv, lat, acc)
        conj (dict): conjunto de soluciones pendientes, indexado por individuo
        tabuSet (set): individuos ya explorados que se excluyen

    Returns:
        None
    '''
    key_new = tuple(sol_new[0])
    if tabuSet is None or key_new not in tabuSet:
        dominated_keys = []
        for k, s in conj.items(): #s = (indv, lat, acc)
            d = dominate(s, sol_new)
            if d > 0:
                return
            elif d < 0:
                dominated_keys.append(k)
        for k in dominated_keys:
            del conj[k]

        conj[key_new] = sol_new

def VNS(maxEvals: int, maxNeigh: int, search_space: dict, acc_predictor, latency_table, n_part: int = 9) -> list:
    '''
    Ejecuta el algoritmo propuesto hasta agotar el presupuesto de evaluaciones. En cada iteración toma como solución 
    actual la del nicho menos poblado de las pendientes, explora su vecindad por mutación y acepta la mutada si no es dominada.
    Si no quedan pendientes, reinicia desde una solución aleatoria.

    Args:
        maxEvals (int): presupuesto total de evaluaciones
        maxNeigh (int): vecinos sin mejora antes de abandonar la solución actual
        search_space (dict): posibles valores validos para cada hp
        acc_predictor: predictor de accuracy de OFA
        latency_table: tabla de latencia del dispositivo
        n_part (int): particiones de das_dennis para los puntos de referencia

    Returns:
        list: frente de Pareto final, como tuplas (indv, lat, acc)
    '''

    savedSols, n_eval = init_pob(search_space, acc_predictor, latency_table)
    no_explored = dict(savedSols)
    tabuSet = set()     

    ref_points = das_dennis(n_part, 2)

    while n_eval < maxEvals:
        if not no_explored:
            indv = iniSol(search_space)
            lat, acc = calculateAccLat(indv, acc_predictor, latency_table)
            sol = indv, lat, acc
            n_eval += 1

            key = tuple(sol[0])
            try_add_pending(sol, savedSols)
            tabuSet.add(key)
        else:
            sol = get_best_solution(list(no_explored.values()), ref_points)
            key = tuple(sol[0])
            tabuSet.add(key)
            del no_explored[key]

        neighCount = 0
        while neighCount < maxNeigh and n_eval < maxEvals:
            #indv_mutada = newMutSol(sol[0], search_space, maxNeigh, neighCount +1)
            indv_mutada = mutSol(sol[0], search_space)
            lat, acc = calculateAccLat(indv_mutada, acc_predictor, latency_table)
            sol_mutada = indv_mutada, lat, acc
            n_eval += 1

            d = dominate(sol, sol_mutada)
            if d <= 0:                       # la mutada es mejor o igual
                try_add_pending(sol_mutada, savedSols)
                try_add_pending(sol_mutada, no_explored, tabuSet)
                sol = sol_mutada
                neighCount = 0 if d < 0 else neighCount + 1

            else:                           # la mutada es peor
                neighCount += 1


    return list(savedSols.values())

