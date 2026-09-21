'''
Archivo que implementa el análisis de sensibilidad OFAT del espacio de búsqueda. Se genera el producto cartesiano de los cuatro hiperparámetros, 
tomando cada uno un valor global en todas las capas, lo que da un total de 135 arquitecturas. Para cada valor de cada hiperparámetro se
agrupan las arquitecturas que lo contienen, 45 en el caso de ks, e y d, y 27 en el de r, y se resumen mediante su mediana.

Junto al OFAT se calcula la eficiencia accuracy/latencia por valor y el ratio rendimiento/coste (Δacc/Δlat) entre valores consecutivos.
'''

import itertools
import numpy as np
import matplotlib.pyplot as plt
from models_comparison import config_device, config_predictors, config_hiperparameters


def build_full_factorial(search_space):
    '''
    Genera todas las combinaciones posibles tratando cada parámetro como un valor global (mismo valor en todas las capas/etapas del individuo),
    lo que da un total de 135 arquitecturas.

    Args:
        search_space (dict): posibles valores validos para cada hp

    Returns:
        tuple: (list) arquitecturas expandidas como dicts con las 20 posiciones de ks, las 20 de e, las 5 de d y la resolución;
               (list) valores globales usados en cada arquitectura
    '''
    ks_opts = search_space["ks"]
    e_opts  = search_space["e"]
    d_opts  = search_space["d"]
    r_opts  = search_space["r"]

    architectures = []
    metadata = []

    for ks_val, e_val, d_val, r_val in itertools.product(
        ks_opts, e_opts, d_opts, r_opts
    ):
        arch = {
            "ks": [ks_val] * 20,
            "e":  [e_val]  * 20,
            "d":  [d_val]  * 5,
            "r":  [r_val],     
        }
        architectures.append(arch)
        metadata.append({
            "ks": ks_val,
            "e":  e_val,
            "d":  d_val,
            "r":  r_val,
        })

    print(f"Total de arquitecturas generadas: {len(architectures)}")
    return architectures, metadata


def evaluate_architectures(architectures, acc_predictor, efficiency_predictor):
    '''
    Evalúa cada arquitectura con el predictor de accuracy y la tabla de latencia, manteniendo el orden de la lista de entrada.

    Args:
        architectures (list): arquitecturas expandidas
        acc_predictor: predictor de accuracy de OFA
        efficiency_predictor: tabla de latencia del dispositivo

    Returns:
        tuple: (ndarray) accuracies, (ndarray) latencias
    '''
    accuracies = []
    latencies  = []

    for arch in architectures:
        acc = acc_predictor.predict_accuracy([arch])[0]
        lat = efficiency_predictor.predict_efficiency([arch])[0]
        accuracies.append(acc)
        latencies.append(lat)

    return np.array(accuracies), np.array(latencies)


def ofat_analysis(metadata, accuracies, latencies, search_space):
    '''
    Para cada parámetro y cada uno de sus valores posibles, agrupa todas las arquitecturas donde ese parámetro tiene ese valor.
    El tamaño de cada grupo es 135 entre el número de valores del parámetro, es decir, 45 para ks, e y d, y 27 para r.

    Devuelve un dict con la estructura:
    {
      "ks": { 3: {"acc": [...], "lat": [...], "n": 45},
              5: {...},
              7: {...} },
      "e":  { ... },
      ...
    }

    Args:
        metadata (list): valores globales de cada arquitectura
        accuracies (ndarray): accuracy de cada arquitectura
        latencies (ndarray): latencia de cada arquitectura
        search_space (dict): posibles valores validos para cada hp

    Returns:
        dict: grupos de acc y lat por parámetro y valor
    '''

    results = {}

    for param_name, values in search_space.items():
        results[param_name] = {}

        for fixed_value in values:
            indices = [
                i for i, m in enumerate(metadata)
                if m[param_name] == fixed_value
            ]

            results[param_name][fixed_value] = {
                "acc": accuracies[indices],
                "lat": latencies[indices],
                "n":   len(indices)
            }

    return results

def plot_ofat(ofat_results):
    '''
    Representa el análisis OFAT en dos figuras independientes, una por objetivo, cada una con una rejilla 2x2 que recoge
    los boxplots de los cuatro hiperparámetros y anota la mediana sobre cada caja.

    Guarda las figuras en ofat_accuracy.png y ofat_latencia.png.

    Args:
        ofat_results (dict): results[param][valor] -> {"acc": ndarray, "lat": ndarray, "n": int}

    Returns:
        None

    Nota: se ha utilizado Claude Opus 5 como apoyo en la generación de esta función.
    '''
    params = list(ofat_results.keys())
    colors = ["#B5D4F4", "#5DCAA5", "#F0997B", "#AFA9EC", "#FAC775"]

    objectives = [
        ("acc", "Accuracy", "ofat_accuracy.png"),
        ("lat", "Latencia [ms]",       "ofat_latencia.png"),
    ]

    for obj_key, obj_label, filename in objectives:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7), sharey=True)
        fig.suptitle(
            f"Análisis OFAT: efecto de cada parámetro sobre {obj_label}",
            fontsize=12
        )

        for idx, param_name in enumerate(params):
            ax = axes[idx // 2, idx % 2]
            param_data = ofat_results[param_name]
            values = sorted(param_data.keys())

            data   = [np.array(param_data[v][obj_key]).flatten() for v in values]
            labels = [str(v) for v in values]

            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f"{param_name} → {obj_label}", fontsize=10)
            ax.set_xlabel(f"Valor de {param_name}")
            if idx % 2 == 0:
                ax.set_ylabel(obj_label)
            ax.grid(True, linestyle=":", alpha=0.6)

            for i, d in enumerate(data):
                ax.text(
                    i + 1, np.median(d),
                    f"{np.median(d):.1f}",
                    ha="center", va="bottom",
                    fontsize=8, color="navy"
                )

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.show()



def plot_acc_cost_ratio_ofat(ofat_results):
    '''
    Representa, en una rejilla 2x2, los boxplots del ratio accuracy/latencia por valor de cada hiperparámetro.
    El ratio se calcula arquitectura a arquitectura antes de agrupar, anotando la mediana sobre cada caja.

    Guarda la figura en eficiencia_acc_lat.png.

    Args:
        ofat_results (dict): results[param][valor] -> {"acc": ndarray, "lat": ndarray, "n": int}

    Returns:
        None

    Nota: se ha utilizado Claude Opus 5 como apoyo en la generación de esta función.
    '''
    params = list(ofat_results.keys())
    colors = ["#B5D4F4", "#5DCAA5", "#F0997B", "#AFA9EC", "#FAC775"]

    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(9, 5.5),
        sharey=True,
    )
    fig.suptitle(
        "Eficiencia accuracy/latencia por valor de cada hiperparámetro",
        fontsize=12,
    )

    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        param_data = ofat_results[param]
        values = sorted(param_data.keys())

        # Ratio por arquitectura: acc / lat
        data = [
            np.asarray(param_data[v]["acc"]).flatten() /
            np.asarray(param_data[v]["lat"]).flatten()
            for v in values
        ]
        labels = [str(v) for v in values]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"{param} → acc/lat", fontsize=10)
        ax.set_xlabel(f"Valor de {param}")
        if idx % 2 == 0:
            ax.set_ylabel("Accuracy / Latencia  (pp por ms)")
        ax.grid(True, linestyle=":", alpha=0.6)

        for i, d in enumerate(data):
            ax.text(
                i + 1, np.median(d),
                f"{np.median(d):.2f}",
                ha="center", va="bottom",
                fontsize=8, color="navy",
            )

    plt.tight_layout()
    plt.savefig("eficiencia_acc_lat.png", dpi=150)
    plt.show()


def performance_cost_ratio(ofat_results):
    '''
    Calcula el ratio Δacc/Δlat entre valores consecutivos (marginal) y la pendiente global (extremo a extremo) de cada hiperparámetro,
    usando medianas para mitigar el efecto de los outliers de latencia. Imprime la tabla de resultados por consola.

    Args:
        ofat_results (dict): results[param][valor] -> {"acc": ndarray, "lat": ndarray, "n": int}

    Returns:
        tuple: (dict) transiciones por parámetro, (dict) pendiente global por parámetro
    '''

    marginal = {}
    global_slope = {}

    for param, levels in ofat_results.items():
        values = sorted(levels.keys())
        med_acc = [np.median(levels[v]["acc"]) for v in values]
        med_lat = [np.median(levels[v]["lat"]) for v in values]

        rows = []
        for i in range(len(values) - 1):
            d_acc = med_acc[i + 1] - med_acc[i]
            d_lat = med_lat[i + 1] - med_lat[i]
            rows.append({
                "from": values[i], "to": values[i + 1],
                "d_acc": d_acc, "d_lat": d_lat,
                "ratio": d_acc / d_lat if d_lat != 0 else np.nan,
                "lat_from": med_lat[i], "lat_to": med_lat[i + 1],
                "acc_from": med_acc[i], "acc_to": med_acc[i + 1],
            })
        marginal[param] = rows
        global_slope[param] = (
            (med_acc[-1] - med_acc[0]) / (med_lat[-1] - med_lat[0])
            if med_lat[-1] != med_lat[0] else np.nan
        )

    print(f"\n{'Param':<6}{'Transición':<16}{'Δacc (pp)':<12}"
          f"{'Δlat (ms)':<12}{'Δacc/Δlat':<12}")
    print("-" * 58)
    for param, transitions in marginal.items():
        for t in transitions:
            label = f"{t['from']} → {t['to']}"
            print(f"{param:<6}{label:<16}{t['d_acc']:<12.3f}"
                  f"{t['d_lat']:<12.3f}{t['ratio']:<12.4f}")
        print()
    print("Sensibilidad global (pp por ms, rango completo):")
    for param, slope in global_slope.items():
        print(f"  {param}: {slope:.4f}")

    return marginal, global_slope


def plot_performance_cost_ratio(ofat_results):
    '''
    Visualiza el ratio rendimiento/coste en dos figuras de 2x2 paneles. La primera recoge el Δacc/Δlat de cada transición 
    en un diagrama de barras y la segunda la trayectoria acc-lat de cada hiperparámetro, con el ratio anotado sobre cada
    segmento y escalas compartidas para que las pendientes sean comparables.
    
    Guarda las figuras en ratio_marginal.png y trayectoria_acc_lat.png.

    Args:
        ofat_results (dict): results[param][valor] -> {"acc": ndarray, "lat": ndarray, "n": int}

    Returns:
        None

    Nota: se ha utilizado Claude Opus 5 como apoyo en la generación de esta función.
    '''
    marginal, global_slope = performance_cost_ratio(ofat_results)

    params = list(ofat_results.keys())
    colors = ["#B5D4F4", "#5DCAA5", "#F0997B", "#AFA9EC", "#FAC775"]

    all_ratios = [t["ratio"] for rows in marginal.values() for t in rows]
    y_max_ratio = max(all_ratios) * 1.20

    all_acc, all_lat = [], []
    for p in params:
        all_acc.extend(np.median(ofat_results[p][v]["acc"]) for v in ofat_results[p])
        all_lat.extend(np.median(ofat_results[p][v]["lat"]) for v in ofat_results[p])
    acc_margin = (max(all_acc) - min(all_acc)) * 0.05
    lat_margin = (max(all_lat) - min(all_lat)) * 0.05
    acc_lim = (min(all_acc) - acc_margin * 3, max(all_acc) + acc_margin)
    lat_lim = (min(all_lat) - lat_margin, max(all_lat) + lat_margin)

    # ratio marginal por transición
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5.5), sharey=True)
    fig.suptitle(
        "Análisis rendimiento/coste: ganancia de accuracy por ms añadido (Δacc/Δlat)",
        fontsize=12
    )

    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        rows = marginal[param]
        labels = [f"{t['from']}→{t['to']}" for t in rows]
        ratios = [t["ratio"] for t in rows]

        bars = ax.bar(
            labels, ratios,
            color=colors[: len(rows)],
            edgecolor="#444", linewidth=0.8, alpha=0.7,
        )

        ax.set_title(
            f"{param} → Δacc/Δlat  (global: {global_slope[param]:.3f})",
            fontsize=10
        )
        ax.set_xlabel(f"Transición de {param}")
        if idx % 2 == 0:
            ax.set_ylabel("Δacc / Δlat  (pp por ms)")
        ax.set_ylim(0, y_max_ratio)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)

        for bar, r in zip(bars, ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max_ratio * 0.02,
                f"{r:.3f}",
                ha="center", va="bottom",
                fontsize=8, color="navy",
            )

    plt.tight_layout()
    plt.savefig("ratio_marginal.png", dpi=150)
    plt.show()

    # trayectoria acc-lat
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5.5), sharex=True, sharey=True)
    fig.suptitle(
        "Trayectoria accuracy-latencia por hiperparámetro",
        fontsize=12
    )

    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        values = sorted(ofat_results[param].keys())
        med_acc = [np.median(ofat_results[param][v]["acc"]) for v in values]
        med_lat = [np.median(ofat_results[param][v]["lat"]) for v in values]

        ax.plot(
            med_lat, med_acc, "-o",
            color=colors[idx], linewidth=2.2, markersize=9,
            markeredgecolor="#333",
        )

        for v, lat, acc in zip(values, med_lat, med_acc):
            ax.annotate(
                str(v), xy=(lat, acc),
                xytext=(6, -12), textcoords="offset points",
                fontsize=8, color="#333", fontweight="bold",
            )

        for t in marginal[param]:
            x_mid = (t["lat_from"] + t["lat_to"]) / 2
            y_mid = (t["acc_from"] + t["acc_to"]) / 2
            ax.annotate(
                f"{t['ratio']:.3f}",
                xy=(x_mid, y_mid),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color="navy",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="white", ec=colors[idx],
                    alpha=0.9, linewidth=1,
                ),
            )

        ax.set_title(f"{param} → trayectoria acc-lat", fontsize=10)
        ax.set_xlabel("Latencia mediana [ms]")
        if idx % 2 == 0:
            ax.set_ylabel("Accuracy mediana (%)")
        ax.set_xlim(lat_lim)
        ax.set_ylim(acc_lim)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("trayectoria_acc_lat.png", dpi=150)
    plt.show()



def main_ofat():
    '''
    Flujo completo del análisis OFAT:
    1. Configurar el dispositivo, los predictores y el espacio de búsqueda.
    2. Generar el factorial completo de arquitecturas con valores globales.
    3. Evaluar cada arquitectura con los predictores de accuracy y latencia.
    4. Realizar el análisis OFAT agrupando por cada parámetro y valor.
    5. Visualizar los resultados con boxplots para acc y latencia.
    6. Calcular y visualizar la eficiencia acc/latencia por parámetro.
    7. Calcular y visualizar el ratio rendimiento/coste entre valores consecutivos.

    Returns:
        tuple: (dict) resultados OFAT, (list) metadatos, (ndarray) accuracies,
            (ndarray) latencias
    '''
    device = config_device()
    acc_predictor, latency_table, _ = config_predictors(device)
    _, _, _, search_space = config_hiperparameters()

    print("=== Generando factorial completo ===")
    architectures, metadata = build_full_factorial(search_space)

    print("\n=== Evaluando arquitecturas ===")
    accuracies, latencies = evaluate_architectures(
        architectures, acc_predictor, latency_table
    )

    print("\n=== Análisis OFAT ===")
    ofat_results = ofat_analysis(metadata, accuracies, latencies, search_space)

    print("\n=== Visualizando OFAT ===")
    plot_ofat(ofat_results)

    print("\n=== Eficiencia acc/lat ===")
    plot_acc_cost_ratio_ofat(ofat_results)

    print("\n=== Análisis rendimiento/coste ===")
    plot_performance_cost_ratio(ofat_results)

    return ofat_results, metadata, accuracies, latencies


if __name__ == "__main__":
    ofat_results, metadata, accuracies, latencies = main_ofat()
