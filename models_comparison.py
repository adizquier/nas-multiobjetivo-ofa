# general
import time
import random
import pickle

# AI/ML/NN
import torch
import numpy as np

# Matplotlib
from matplotlib import pyplot as plt

# OFA/OFA²
from ofa2.tutorial.accuracy_predictor import AccuracyPredictor
from ofa2.tutorial.latency_table import LatencyTable
from ofa2.tutorial.flops_table import FLOPsTable

# pymoo
from pymoo.termination import get_termination
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.population import Population

'''
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.moo.sms import cv_and_dom_tournament
'''

from ofa_problem import OfaProblem, OfaSampling
from operadores import OfaMinCrossover, OfaGuidedMutation, OfaBiasedCrossover
from ofa_problem import OfaProblemPSO, OfaSamplingPSO
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.crossover.ux import UniformCrossover

import VNS


def config_device():
    '''
    Selecciona el dispositivo de ejecución, usando la GPU si está disponible y la CPU en caso contrario.
    Con GPU activa cuDNN y su modo benchmark, que selecciona los algoritmos de convolución más rápidos para los tamaños de entrada empleados.

    Returns:
        str: identificador del dispositivo, "cuda:0" o "cpu"
    '''

    cuda_available = torch.cuda.is_available()
    device = "cuda:0" if cuda_available else "cpu"
    if cuda_available:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    return device


def config_predictors(device):
    '''
    Carga el predictor de accuracy preentrenado de OFA, la tabla de latencia del dispositivo objetivo y la tabla de FLOPs.

    Args:
        device (str): dispositivo de ejecución de los predictores

    Returns:
        tuple: (AccuracyPredictor) predictor de accuracy, (LatencyTable) tabla de
            latencia, (FLOPsTable) tabla de FLOPs
    '''

    target_hardware = "note10"

    print("Cargando predictores de precisión y latencia...")
    acc_predictor = AccuracyPredictor(pretrained=True, device=device)
    latency_table = LatencyTable(device=target_hardware)
    flops_table = FLOPsTable(device=device)
    print("Predictores cargados.")

    return acc_predictor, latency_table, flops_table


def config_hiperparameters():
    '''
    Define los hiperparámetros de ejecución de los algoritmos y el espacio de búsqueda de OFA.
    El producto de población y generaciones fija el presupuesto de 100.000 evaluaciones que comparten todos los algoritmos.

    Returns:
        tuple: (int) tamaño de población
               (int) numero de generaciones
               (float) probabilidad de mutación
               (dict) valores validos de tamaño de kernel, ratio de expansión, profundidad y resolución
    '''

    POPULATION_SIZE = 100
    GENERATIONS = 1000
    MUTATION_PROB = 0.1

    search_space = {
        'ks': [3, 5, 7],
        'e': [3, 4, 6],
        'd': [2, 3, 4],
        'r': [160, 176, 192, 208, 224]
    }

    return POPULATION_SIZE, GENERATIONS, MUTATION_PROB, search_space

def config_algorithm(POPULATION_SIZE, MUTATION_PROB):
    '''
    Construye los tres algoritmos competidores. NSGA-II y SMS-EMOA emplean los operadores genéricos de pymoo, mientras que SPEA2 usa los operadores propios desarrollados para
    el espacio de OFA.

    Args:
        POPULATION_SIZE (int): tamaño de población
        MUTATION_PROB (float): probabilidad de mutación

    Returns:
        tuple: (NSGA2) NSGA-II, (SMSEMOA) SMS-EMOA, (SPEA2) SPEA2
    '''
    sampling = OfaSampling()
    crossover = UniformCrossover()
    mutation = ChoiceRandomMutation(prob=MUTATION_PROB)

    spea_crossover = OfaBiasedCrossover(p_min=0.7)
    spea_mutation = OfaGuidedMutation(prob=MUTATION_PROB)

    algorithm_nsga = NSGA2(
        pop_size=POPULATION_SIZE,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
    )
    algorithm_smsemoa = SMSEMOA(
        pop_size=POPULATION_SIZE,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
    )
    algorithm_spea2 = SPEA2(
        pop_size=POPULATION_SIZE,
        sampling=sampling,
        crossover=spea_crossover,
        mutation=spea_mutation,
    )

    '''
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
    sel=TournamentSelection(func_comp=cv_and_dom_tournament)

    algorithm_rvea = RVEA(
        pop_size=POPULATION_SIZE,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        ref_dirs=ref_dirs,
        selection=sel,
    )
    algorithm_ctaea = CTAEA(
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
    )'''

    return algorithm_nsga, algorithm_smsemoa, algorithm_spea2


def random_seeds(numero_de_semillas=31):
    '''
    Genera las semillas de las ejecuciones independientes a partir de una semilla maestra fija, de modo que todos los algoritmos se ejecutan
    sobre el mismo conjunto de semillas y el experimento es reproducible.

    Args:
        numero_de_semillas (int): numero de ejecuciones independientes

    Returns:
        list: semillas generadas
    '''

    random.seed(12345)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(numero_de_semillas)]

    return seeds


def run_algorithm(algorithm, problem, GENERATIONS, seeds, name_algorithm):
    '''
    Ejecuta un algoritmo una vez por semilla, fijando los generadores aleatorios antes de cada ejecución para que sea reproducible,
    y devuelve la población final de cada una ordenada por latencia creciente junto con su tiempo de ejecución.

    Args:
        algorithm: algoritmo de pymoo a ejecutar
        problem (OfaProblem): problema multiobjetivo sobre el espacio de OFA
        GENERATIONS (int): numero de generaciones de cada ejecución
        seeds (list): semillas de las ejecuciones independientes
        name_algorithm (str): nombre del algoritmo, usado en las trazas por consola

    Returns:
        list: un dict por ejecución con las claves indice, seed, population y time
    '''

    results_execution = []
    for iteracion, seed in enumerate(seeds):
        print(f"Running algorithm {name_algorithm} en iteracion {iteracion} with seed {seed}...")
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        termination = get_termination("n_gen", GENERATIONS)

        start_time = time.time()
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Semilla {seed} en iteracion {iteracion} completada en {elapsed_time:.2f} segundos para el algoritmo {name_algorithm}.")

        lat_values = result.pop.get("F")[:, 0]
        idx = np.argsort(lat_values)
        res_reordered = result.pop[idx]

        results_execution.append({
            "indice": iteracion,
            "seed": seed,
            "population": res_reordered,
            "time": elapsed_time
        })

    return results_execution


def run_all_algorithms(problem, POPULATION_SIZE, GENERATIONS, MUTATION_PROB, seeds, suffix):
    '''
    Ejecuta los tres algoritmos competidores sobre las semillas indicadas y guarda los resultados de cada uno en new_results,
    siguiendo la convención de nombres que espera cargar_series.

    Args:
        problem (OfaProblem): problema multiobjetivo sobre el espacio de OFA
        POPULATION_SIZE (int): tamaño de población
        GENERATIONS (int): numero de generaciones de cada ejecución
        MUTATION_PROB (float): probabilidad de mutación
        seeds (list): semillas de las ejecuciones independientes
        suffix (str): sufijo del archivo, lat para latencia y FLOPS para FLOPs

    Returns:
        dict: resultados por algoritmo, con la lista de sus ejecuciones
    '''
    algorithms = config_algorithm(POPULATION_SIZE, MUTATION_PROB)
    all_results = {}

    for algorithm, name in zip(algorithms, ["NSGA2", "SMSEMOA", "SPEA2"]):
        path = f"new_results/{name}_results_{suffix}.pkl"
        print(f"Ejecutando {name}...")
        results = run_algorithm(algorithm, problem, GENERATIONS, seeds, name)
        save_results(results, path)
        print(f"Guardado: {path}")
        all_results[name] = results

    return all_results

def run_pso():
    '''
    Ejecuta CMOPSO y MOPSO-CD sobre las 31 semillas con el mismo presupuesto de evaluaciones
    que el resto de competidores. Ambos operan sobre OfaProblemPSO, que codifica cada
    hiperparámetro como un índice entero acotado y lo decodifica al valor correspondiente
    de su dominio.

    Returns:
        None
    '''

    device = config_device()
    acc_predictor, latency_table, _ = config_predictors(device)
    POPULATION_SIZE, GENERATIONS, _, search_space = config_hiperparameters()

    problem = OfaProblemPSO(
        efficiency_predictor=latency_table,
        accuracy_predictor=acc_predictor,
        search_vars=search_space,
    )

    seeds = random_seeds(31)

    algorithms = [
        CMOPSO(pop_size=POPULATION_SIZE, sampling=OfaSamplingPSO()),
        MOPSO_CD(pop_size=POPULATION_SIZE, sampling=OfaSamplingPSO()),
    ]

    for algorithm, name in zip(algorithms, ["CMOPSO", "MOPSO_CD"]):
        path = f"new_results/{name}_results_lat.pkl"
        print(f"Ejecutando {name}...")
        results = run_algorithm(algorithm, problem, GENERATIONS, seeds, name)
        save_results(results, path)
        print(f"Guardado: {path}")


def save_results(results, filename):
    '''
    Guarda los resultados de las ejecuciones en un archivo binario mediante pickle.

    Args:
        results (list): resultados devueltos por run_algorithm
        filename (str): ruta del archivo de destino

    Returns:
        None
    '''

    with open(filename, "wb") as f:
        pickle.dump(results, f)


def load_results(filename):
    '''
    Carga los resultados de ejecuciones previas guardados con save_results.

    Args:
        filename (str): ruta del archivo a cargar

    Returns:
        list: resultados de las ejecuciones
    '''

    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


def return_min_max_values(all_results, objective_index=0):
    '''
    Recorre los resultados de todos los algoritmos y ejecuciones para obtener el mínimo y el máximo de un objetivo, que permite normalizar
    conjuntamente los algoritmos de modo que sus hipervolúmenes puedan compararse.

    Args:
        all_results (dict): resultados por algoritmo, con la lista de sus ejecuciones
        objective_index (int): objetivo del que se obtienen los extremos, 0 para latencia y 1 para accuracy

    Returns:
        tuple: (float) mínimo, (float) máximo
    '''

    max_val, min_val = float('-inf'), float('inf')

    for runs_list in all_results.values():
        for run in runs_list:
            vals = run["population"].get("F")[:, objective_index]
            max_val = max(max_val, np.max(vals))
            min_val = min(min_val, np.min(vals))

    return min_val, max_val


def norm_results(all_results):
    '''
    Normaliza al rango [0, 1] los dos objetivos de todas las ejecuciones, empleando como extremos los valores mínimo y máximo de todos los algoritmos en conjunto. 
    Modifica all_results.

    Args:
        all_results (dict): resultados por algoritmo, con la lista de sus ejecuciones

    Returns:
        None
    '''

    min_val_lat, max_val_lat = return_min_max_values(all_results, objective_index=0)
    min_val_err, max_val_err = return_min_max_values(all_results, objective_index=1)

    for runs_list in all_results.values():
        for run in runs_list:
            F = run["population"].get("F")

            denom_lat = (max_val_lat - min_val_lat) if max_val_lat != min_val_lat else 1.0
            denom_err = (max_val_err - min_val_err) if max_val_err != min_val_err else 1.0

            F[:, 0] = (F[:, 0] - min_val_lat) / denom_lat
            F[:, 1] = (F[:, 1] - min_val_err) / denom_err

            run["population"].set("F", F)


def calc_hiperv(result):
    '''
    Calcula el hipervolumen de la población final de una ejecución, tomando como punto de referencia [1.2, 1.2].
    Los objetivos deben estar normalizados previamente.

    Args:
        result (dict): una ejecución, con su población en la clave population

    Returns:
        float: hipervolumen de la población
    '''

    population = result["population"]
    ref_point = np.array([1.2, 1.2])

    ind = HV(ref_point=ref_point)
    hv = ind(population.get("F"))
    return hv


def hipervolumen_all_algorithms(all_results):
    '''
    Calcula el hipervolumen de todas las ejecuciones de cada algoritmo.

    Args:
        all_results (dict): resultados por algoritmo, con la lista de sus ejecuciones

    Returns:
        dict: hv_results[algoritmo] -> lista con el hipervolumen de cada ejecución
    '''

    hv_results = {}

    for name, results in all_results.items():
        hv_list = []
        for result in results:
            hv = calc_hiperv(result)
            hv_list.append(hv)
        hv_results[name] = hv_list

    return hv_results


def select_median_population(all_results):
    '''
    Selecciona, para cada algoritmo, la ejecución cuyo hipervolumen es el más próximo a la mediana de sus ejecuciones.
    Sustituye en all_results la lista de ejecuciones por la seleccionada.

    Args:
        all_results (dict): resultados por algoritmo, con la lista de sus ejecuciones

    Returns:
        dict: indices_to_keep[algoritmo] -> índice de la ejecución seleccionada
    '''

    hiperv = hipervolumen_all_algorithms(all_results)
    indices_to_keep = {}

    for name, hv_list in hiperv.items():
        hv_values = np.array(hv_list)

        target_median = np.median(hv_values)
        median_index = np.argmin(np.abs(hv_values - target_median))

        all_results[name] = all_results[name][median_index]
        indices_to_keep[name] = median_index

    return indices_to_keep


def plot_results(all_results):
    '''
    Dibuja los frentes de Pareto de las corridas medianas de cada algoritmo, con la latencia normalizada en el eje X 
    y la accuracy normalizada en el Y. Guarda la figura en frentes_pareto.png.

    Args:
        all_results (dict): resultados por algoritmo, ya reducidos a su corrida mediana

    Returns:
        None

    Nota: se ha utilizado Claude Opus 5 como apoyo en la generación de esta función.
    '''

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    markers = {
        "NSGA2": "o",
        "SMSEMOA": "s",
        "SPEA2": "^",
        "RVEA": "D",
        "CTAEA": "v",
        "CMOPSO": "P",
        "MOPSO_CD": "X",
        "SPEA2_prob": "p",
        "SPEA2_min": "h",
        "SPEA2_old": "<",
    }
    labels = {
        "SPEA2_prob": "SPEA2 cruce con probabilidad",
        "SPEA2_min": "SPEA2 cruce sin probabilidad",
        "SPEA2_old": "SPEA2 anterior",
    }
    for name, result in all_results.items():
        population = result["population"]
        F = population.get("F")
        ax.scatter(
            F[:, 0], 1 - F[:, 1],
            label=labels.get(name, name),
            marker=markers.get(name, "o"),
            s=45, alpha=0.7,
        )

    ax.set_xlabel("Latencia normalizada", fontsize=19, labelpad=12)
    ax.set_ylabel("Accuracy normalizada", fontsize=19, labelpad=12)
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(loc="lower right", fontsize=17, markerscale=2.0, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.savefig("frentes_pareto.png", dpi=300)
    plt.show()


def run_spea(run_name):
    '''
    Ejecuta la configuración actual de SPEA2 sobre las 31 semillas, almacenando los resultados. Se lanza una vez por variante de cruce.

    Args:
        run_name (str): nombre de la variante, usado en las trazas y en la ruta de salida

    Returns:
        None
    '''

    device = config_device()
    acc_predictor, latency_table, _ = config_predictors(device)
    POPULATION_SIZE, GENERATIONS, MUTATION_PROB, search_space = config_hiperparameters()

    problem = OfaProblem(
        efficiency_predictor=latency_table,
        accuracy_predictor=acc_predictor,
        search_vars=search_space,
    )

    seeds = random_seeds(31)

    algorithm = config_algorithm(POPULATION_SIZE, MUTATION_PROB)[2]
    results = run_algorithm(algorithm, problem, GENERATIONS, seeds, run_name)

    path = f"new_results/{run_name}_results_lat.pkl"
    save_results(results, path)
    print(f"Guardado: {path}")


def run_VNS(acc_predictor, latency_table, max_neigh = 11, n_part = 9):
    '''
    Ejecuta el VNS sobre las 31 semillas con el mismo presupuesto de evaluaciones que los competidores, guardando el frente de Pareto de cada 
    ejecución como una matriz de objetivos.

    Args:
        acc_predictor: predictor de accuracy de OFA
        latency_table: tabla de latencia del dispositivo
        max_neigh (int): numero de estructuras de vecindad del VNS
        n_part (int): particiones de das_dennis para los puntos de referencia

    Returns:
        None
    '''

    POPULATION_SIZE, GENERATIONS, _, search_space = config_hiperparameters()

    seeds = random_seeds(31)
    results = []

    for i, seed in enumerate(seeds):
        time_init = time.time()
        print(f"Ejecutando VNS con la semilla {i}")
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        pf = VNS.VNS(POPULATION_SIZE * GENERATIONS, max_neigh, search_space, acc_predictor, latency_table, n_part)        
        F = np.array([[sol[1], sol[2]] for sol in pf])
        results.append(F)
        elapsed = time.time() - time_init
        print(f"Semilla {i+1} terminada en {elapsed/60:.1f} min")

    save_results(results, f"new_results/VNS/VNS_best.pkl")

def p_sweep():
    '''
    Algoritmo que realizará el recorrido parametrico para max_Neigh y numero de particiones en la ejecucion del algoritmo VNS

    Retruns:
        None
    '''

    device = config_device()
    acc_predictor, latency_table, _ = config_predictors(device)

    print("Comienzo de barrido paramétrico\n")

    for i in range (2, 23, 2):
        print(f"Valor de n_part: {i} \n")
        time_init = time.time()

        run_VNS(acc_predictor, latency_table, max_neigh=16, n_part=i)

        elapsed = time.time() - time_init
        print(f"31 ejecuciones terminadas en {elapsed/60:.1f} min para valor de n_part {i}")


def cargar_barrido(valores=range(2, 23, 2)):
    '''
    Carga los resultados del barrido paramétrico de n_part, envolviendo las matrices de objetivos del VNS en objetos Population 
    para que puedan tratarse igual que los resultados de los algoritmos de pymoo.

    Args:
        valores (iterable): valores de n_part cuyos resultados se cargan

    Returns:
        dict: all_results["n_part_{valor}"] -> lista de ejecuciones
    '''

    all_results = {}
    for value in valores:
        path = f"new_results/VNS/VNS_resultsNewMut_nPart_{value}.pkl"
        all_results[f"n_part_{value}"] = _wrap(load_results(path))
    return all_results

def analizar_barrido():
    '''
    Carga el barrido paramétrico de n_part, normaliza todas las configuraciones en conjunto e imprime la media y la desviación típica de su hipervolumen.

    Returns:
        dict: hv_results["n_part_{valor}"] -> hipervolumen de cada ejecución
    '''

    all_results = cargar_barrido()

    norm_results(all_results)
    hv_results = hipervolumen_all_algorithms(all_results)

    for name, hv_list in hv_results.items():
        arr = np.array(hv_list) / 1.44
        print(f"{name}: media={arr.mean():.4f}  std={arr.std():.4f}")

    return hv_results


def _wrap(raw):
    '''
    Adapta una lista de resultados al formato del pipeline. Los arrays de objetivos del VNS se envuelven en objetos Population y los resultados 
    de pymoo se devuelven tal cual.

    Args:
        raw (list): resultados cargados de un .pkl

    Returns:
        list: ejecuciones en formato uniforme
    '''

    if raw and isinstance(raw[0], np.ndarray):
        return [{"population": Population.new("F", F)} for F in raw]
    return raw


def cargar_series(series, latencia=True):
    '''
    Carga varias series de resultados en un único dict.

    Args:
        series (list): nombres o tuplas (etiqueta, ruta) de las series a cargar
        latencia (bool): True para los resultados con latencia, False para los de FLOPs

    Returns:
        dict: all_results[nombre] -> lista de ejecuciones
    '''

    suffix = "lat" if latencia else "FLOPS"
    all_results = {}

    for serie in series:
        if isinstance(serie, tuple):
            name, path = serie
        else:
            name = serie
            path = f"new_results/{name}_results_{suffix}.pkl"
        print(f"Cargando {name} desde {path} ...")
        all_results[name] = _wrap(load_results(path))

    return all_results


def tabla_hv(all_results, normalizar_hv=True):
    '''
    Imprime la media, la desviación típica y la mediana del hipervolumen de cada serie, ordenadas de mayor a menor media.
    Requiere all_results normalizado con norm_results, ya que los hipervolúmenes solo son comparables entre series normalizadas juntas.

    Args:
        all_results (dict): resultados por serie, ya normalizados
        normalizar_hv (bool): True para expresarlo como fracción del máximo alcanzable

    Returns:
        dict: stats[serie] -> (media, std, mediana)
    '''

    hv_results = hipervolumen_all_algorithms(all_results)
    factor = 1.44 if normalizar_hv else 1.0

    stats = {}
    for name, hv_list in hv_results.items():
        hv = np.array(hv_list) / factor
        stats[name] = (np.mean(hv), np.std(hv, ddof=1), np.median(hv))

    print(f"\n{'Serie':<22}{'Media':>10}{'Std':>10}{'Mediana':>10}")
    for name in sorted(stats, key=lambda n: stats[n][0], reverse=True):
        media, std, mediana = stats[name]
        print(f"{name:<22}{media:>10.4f}{std:>10.4f}{mediana:>10.4f}")

    return stats


def comparar(series, latencia=True, tabla=True, grafica=True):
    '''
    Carga las series, las normaliza en conjunto, imprime la tabla de hipervolumen y dibuja los frentes de las ejecuciones medianas.

    Args:
        series (list): nombres o tuplas (etiqueta, ruta) de las series a comparar
        latencia (bool): True para latencia, False para FLOPs
        tabla (bool): True para imprimir la tabla de hipervolumen
        grafica (bool): True para dibujar los frentes

    Returns:
        None
    '''

    all_results = cargar_series(series, latencia)
    norm_results(all_results)

    if tabla:
        tabla_hv(all_results)

    if grafica:
        indices = select_median_population(all_results)
        print(f"Indices de corrida mediana por serie: {indices}")
        plot_results(all_results)


if __name__ == "__main__":
    # --- comparaciones---
    # comparar(["NSGA2", "SMSEMOA", "SPEA2", ("VNS", "new_results/VNS/VNS_best.pkl")])
    # comparar(
    #     [(f"res_{r}", f"new_results/VNS/VNS_res_{r}.pkl") for r in [160, 176, 192, 208, 224]]
    #     + [("VNS_global", "new_results/VNS/VNS_results_partitions_9.pkl")],
    #     tabla=False,
    # )

    # variantes de SPEA2:
    # comparar([
    #     "NSGA2", "SMSEMOA", "SPEA2",
    #     ("SPEA2_min", "new_results/SPEA2_min_results_lat.pkl"),
    #     ("SPEA2_0.3", "new_results/SPEA2_prob_results_lat.pkl"),
    # ])

    comparar(["NSGA2", "SMSEMOA", "SPEA2", ("VNS", "new_results/VNS/VNS_resultsNewMut_nPart_2.pkl")])
