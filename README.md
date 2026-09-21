# Búsqueda de arquitecturas neuronales multiobjetivo mediante VNS sobre la superred Once-for-All

Código del Trabajo Fin de Máster del Máster en Inteligencia Artificial de la Universidad Internacional de Valencia (VIU).

El trabajo aborda la búsqueda automática de arquitecturas neuronales (NAS) como un problema multiobjetivo sobre la superred de Once-for-All (OFA), optimizando de forma simultánea la precisión de la arquitectura y su latencia en un dispositivo objetivo. La aportación consiste en la adaptación de una metaheurística de trayectoria, *Variable Neighborhood Search* (VNS), a este espacio de búsqueda, y en su comparación con algoritmos poblacionales del estado del arte.

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `ofa_problem.py` | Definición del problema multiobjetivo sobre el espacio de OFA. `OfaProblem` emplea variables categóricas y `OfaProblemPSO` una codificación por índices enteros, necesaria para los algoritmos de enjambre de partículas. Incluye los operadores de muestreo correspondientes. |
| `operadores.py` | Operadores de cruce y mutación desarrollados específicamente para el espacio de OFA. |
| `VNS.py` | Implementación del algoritmo propuesto. |
| `models_comparison.py` | Ejecución de los algoritmos competidores, cálculo del hipervolumen, generación de tablas y figuras comparativas. |
| `analisis_sensibilidad.py` | Análisis de sensibilidad OFAT sobre los cuatro hiperparámetros del espacio de búsqueda. |
| `new_results/` | Resultados de las ejecuciones empleadas en la memoria. |
| `setup.py` | Instalación del paquete `ofa2`, adaptado del repositorio de OFA². |

---

## Instalación

Requiere Python 3.11.

**1. Entorno virtual y dependencias.**

```bash
python -m venv entornoTFM
entornoTFM\Scripts\activate        # Windows
source entornoTFM/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

El archivo `requirements.txt` fija las versiones empleadas en la experimentación. Es necesario respetarlas, en particular la de `pymoo`, porque los resultados almacenados contienen objetos serializados de esa biblioteca y solo pueden cargarse con la misma versión con la que se generaron.

**2. Paquete `ofa2`**, que proporciona el predictor de precisión, la tabla de latencia y las utilidades de codificación de arquitecturas:

```bash
git clone https://github.com/ito-rafael/once-for-all-2.git
cp setup.py once-for-all-2/setup.py
cd once-for-all-2 && pip install . && cd ..
```

### Soporte para GPU

La línea `torch==2.10.0` instala la variante por defecto de PyPI, que puede no incluir soporte CUDA según la plataforma. Para ejecutar sobre GPU conviene instalar PyTorch siguiendo las instrucciones de https://pytorch.org antes de instalar el resto de dependencias.

### Predictores

- **Predictor de precisión.** Se descarga automáticamente la primera vez que se instancia `AccuracyPredictor(pretrained=True)`.
- **Tabla de latencia.** Corresponde al dispositivo Samsung Galaxy Note 10 y está disponible en el repositorio público de OFA: https://github.com/han-cai/files/tree/master/ofa

---

## Configuración experimental

El espacio de búsqueda replica el empleado en OFA², con el fin de mantener la comparabilidad de los resultados:

| Hiperparámetro | Valores | Granularidad | Genes |
|---|---|---|---|
| Tamaño de *kernel* | {3, 5, 7} | por capa | 20 |
| Ratio de expansión | {3, 4, 6} | por capa | 20 |
| Profundidad | {2, 3, 4} | por unidad | 5 |
| Resolución de entrada | {160, 176, 192, 208, 224} | global | 1 |

Cada arquitectura queda descrita por un vector de **46 posiciones**.

Todos los algoritmos comparten el mismo presupuesto de **100.000 evaluaciones** (100 individuos × 1.000 generaciones) y se ejecutan sobre **31 semillas independientes**, generadas de forma determinista a partir de una semilla maestra fija.

Los objetivos son la latencia predicha en milisegundos y el error de clasificación, definido como `100 − accuracy`.

---

## Uso

Las funciones de ejecución se encuentran en `models_comparison.py` y se lanzan desde el bloque principal del archivo.

```python
# Algoritmos poblacionales basados en dominancia e indicador
run_all_algorithms(problem, POPULATION_SIZE, GENERATIONS, MUTATION_PROB, seeds, "lat")

# Algoritmos de enjambre de partículas
run_pso()

# Algoritmo propuesto
run_VNS(acc_predictor, latency_table, max_neigh=11, n_part=9)

# Barrido paramétrico del VNS
p_sweep()
```

Para comparar series ya ejecutadas, sin volver a lanzarlas:

```python
comparar(["NSGA2", "SMSEMOA", "SPEA2", ("VNS", "new_results/VNS/VNS_best.pkl")])
```

`comparar` carga los resultados indicados, normaliza conjuntamente los objetivos, imprime la tabla de hipervolumen y representa los frentes de las ejecuciones medianas.

---

## Resultados

Los resultados de todas las ejecuciones empleadas en la memoria se encuentran en `new_results/`, siguiendo la convención de nombres `{algoritmo}_results_lat.pkl`. Cada archivo contiene una lista con las 31 ejecuciones, y cada una guarda su índice, su semilla, la población final y el tiempo empleado.

Se cargan con:

```python
from models_comparison import load_results
resultados = load_results("new_results/NSGA2_results_lat.pkl")
```

### Sobre el hipervolumen

Los valores de hipervolumen se calculan sobre objetivos normalizados al rango [0, 1] tomando como extremos los valores mínimo y máximo de **todas las series comparadas en conjunto**, con el punto de referencia situado en (1,2; 1,2).

Como consecuencia, **los valores solo son comparables dentro de un mismo conjunto de comparación**. Añadir o retirar una serie modifica los límites de normalización y, con ellos, los valores absolutos de todas las demás.

---

## Reproducibilidad

Los resultados empleados en la memoria son los almacenados en `new_results/`, y es sobre esos archivos sobre los que debe verificarse cualquier cifra del documento.

El código permite reejecutar todos los experimentos. Las semillas se fijan de forma determinista, de modo que cualquier reejecución parte del mismo conjunto de 31 semillas.

**Limitación conocida.** Las ejecuciones de CMOPSO y MOPSO-CD no son deterministas: dos ejecuciones con la misma semilla producen resultados distintos. Una reejecución de estos dos algoritmos, por tanto, produce resultados equivalentes en distribución pero no idénticos a los almacenados. Las estadísticas agregadas sobre las 31 semillas, que son las que se reportan en la memoria, no se ven afectadas de forma apreciable por este comportamiento.

---

## Referencias

El trabajo se apoya en los siguientes artículos:

- Cai, H., Gan, C., Wang, T., Zhang, Z. y Han, S. (2020). *Once-for-All: Train One Network and Specialize It for Efficient Deployment*. ICLR.
- Ito, R. C. y Von Zuben, F. J. (2023). *OFA²: A Multi-Objective Perspective for the Once-for-All Neural Architecture Search*. IJCNN.
- Rodríguez-Bejarano, F. M., Santander-Jiménez, S. y Vega-Rodríguez, M. A. (2025). *A multi-objective metaheuristic approach for generating cell type marker panels from single-cell transcriptomics*. Swarm and Evolutionary Computation, 98, 102108.
- Blank, J. y Deb, K. (2020). *pymoo: Multi-Objective Optimization in Python*. IEEE Access, 8, 89497–89509.

---

## Uso de inteligencia artificial

Se ha empleado Claude Opus 5 como apoyo en la generación de algunas funciones de este repositorio, que fueron posteriormente revisadas y modificadas de forma manual. Cada una de ellas incluye una nota en su documentación que lo indica.

---

## Autor y licencia

Adrián Izquierdo Abril — Máster en Inteligencia Artificial, Universidad Internacional de Valencia (VIU), 2026.

Este repositorio se distribuye bajo licencia MIT. Reutiliza componentes de OFA² y los predictores publicados por los autores de OFA, ambos bajo la misma licencia.
