"""Utils for benchmarks."""

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from math import cos, e, pi
from pathlib import Path
from robin.supply.entities import Supply
from typing import Dict, List, Mapping, Union


def get_rus_revenue(supply: Supply, df: pd.DataFrame) -> Dict[str, float]:
    """
    Get the revenue of each RU.

    Args:
        supply: Supply object.
        df: DataFrame from Robins's output data with columns 'service' and 'price'.

    Returns:
        Mapping[str, float]: RU's revenue.
    """
    services_tsp = {service.id: service.tsp.name for service in supply.services}
    df['tsp'] = df['service'].apply(lambda service_id: services_tsp.get(service_id, np.NaN))
    tsp_revenue = df.groupby('tsp').agg({'price': 'sum'}).to_dict()['price']
    return tsp_revenue


def is_better_solution(
    rus_revenue: Mapping[str, float],
    best_solution: Mapping[str, float]
) -> bool:
    """
    Check if the current solution is better than the best solution.

    Args:
        rus_revenue: Revenue of each RU.
        best_solution: Best solution found so far.

    Returns:
        bool: True if the current solution is better than the best solution, False otherwise.
    """
    if not best_solution:
        return True
    elif len(rus_revenue) > len(best_solution):
        return True
    elif sum([rus_revenue[tsp] > best_solution.get(tsp, -np.inf) for tsp in rus_revenue]) >= len(rus_revenue) // 2:
        return True
    return False


def penalty_function(x: float, k: int) -> float:
    """
    Compute the penalty based on a normalized deviation.

    Args:
        x: Normalized deviation.
        k: Scaling factor.

    Returns:
        Penalty value (float).
    """
    return 1 - e ** (-k * x ** 2) * (0.5 * cos(pi * x) + 0.5)


def sns_box_plot(
    df: pd.DataFrame,
     x_data: str,
     y_data: str,
     title: str,
     x_label: str,
     y_label: str,
     hue: Union[str, None] = None,
     save_path: Union[Path, None] = None,
     fig_size: tuple = (10, 6)
) -> None:
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_title(title, fontweight='bold', fontsize=18)

    # Draw the boxplot and stripplot
    boxplot = sns.boxplot(data=df, x=x_data, y=y_data, hue=hue, dodge=True, zorder=1, boxprops=dict(alpha=.3), ax=ax)
    stripplot = sns.stripplot(data=df, x=x_data, y=y_data, hue=hue, dodge=True, alpha=0.5, zorder=1, ax=ax)

    # Remove the stripplot legend handles
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [handle for handle, label in zip(handles, labels) if 'line' not in str(type(handle))]

    if hue:
        ax.legend(handles=new_handles, title=hue, fontsize=12, title_fontsize=14)

    ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    for spn in ('top', 'right', 'bottom', 'left'):
        ax.spines[spn].set_visible(True)
        ax.spines[spn].set_linewidth(1.0)
        ax.spines[spn].set_color('#A9A9A9')

    plt.show()
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)


def sns_line_plot(
    df: pd.DataFrame,
    x_data: str,
    y_data: str,
    title: str,
    x_label: str,
    y_label: str,
    hue: Union[str, None] = None,
    save_path: Union[Path, None] = None,
    legend_type: str = "outside",
    x_limit: tuple = (-1, 100),
    y_limit: tuple = (-1, 4000),
    fig_size: tuple = (10, 6)
) -> None:
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_title(title, fontweight='bold', fontsize=18)
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    sns.lineplot(ax=ax,
                 data=df,
                 x=x_data,
                 y=y_data,
                 hue=hue,
                 legend=True)

    ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    for spn in ('top', 'right', 'bottom', 'left'):
        ax.spines[spn].set_visible(True)
        ax.spines[spn].set_linewidth(1.0)
        ax.spines[spn].set_color('#A9A9A9')

    if legend_type == 'outside':
        plt.legend(
            loc='upper center',  # Base de la posición (arriba y centrada)
            bbox_to_anchor=(0.5, -0.2),  # Desplazamiento debajo del área de la gráfica
            ncol=2,  # Organiza la leyenda en dos columnas
            frameon=True,  # Muestra el marco de la leyenda (opcional)
        )
    else:
        # Inside, bottom right
        plt.legend(loc='lower right')

    plt.tight_layout(rect=[0, 0.15, 1, 1])  # Ajusta los márgenes para incluir la leyenda debajo

    plt.show()
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)


def get_schedule_from_supply(path: Union[Path, None] = None,
                             supply: Union[Supply, None] = None
                             ) -> Mapping[str, Mapping[str, List[int]]]:
    if not supply:
        supply = Supply.from_yaml(path=path)
    requested_schedule = {}
    for service in supply.services:
        requested_schedule[service.id] = {}
        time = service.time_slot.start
        delta = time.total_seconds() // 60
        for stop in service.line.timetable:
            arrival_time = delta + int(service.line.timetable[stop][0])
            departure_time = delta + int(service.line.timetable[stop][1])
            requested_schedule[service.id][stop] = [arrival_time, departure_time]

    return requested_schedule

def plot_scheduled_services(
        fair_index: str,
        summary_df: pd.DataFrame,
        df_history: pd.DataFrame,
        revenue_behavior: Mapping[str, Mapping[str, Union[str, float]]],
        save_path: str,
) -> None:
    # Obtén los índices de las filas donde 'Inequity' es mínima para cada 'FairIndex'
    idx_min = summary_df.groupby('FairIndex')['Inequity'].idxmin()

    # Selecciona las filas correspondientes y extrae las columnas 'FairIndex' y 'Run'
    resultado = summary_df.loc[idx_min, ['FairIndex', 'Run']]

    # Si df_history tiene duplicados para las combinaciones, se agrupan y se toma el primer valor
    df_history_unique = df_history.groupby(['FairIndex', 'Run'], as_index=False)['Discrete'].first()

    # Realiza el merge tipo left para que resultado_final tenga las mismas filas que resultado
    resultado_final = resultado.merge(
        df_history_unique[['FairIndex', 'Run', 'Discrete']],
        on=['FairIndex', 'Run'],
        how='left'
    )

    for row in resultado_final.iterrows():
        if row[1]['FairIndex'] == fair_index:
            scheduled_services = row[1]['Discrete']

    scheduled_by_importance = {}

    for i, t in zip(scheduled_services, revenue_behavior.items()):
        service_id, service = t
        if service['ru'] not in scheduled_by_importance:
            scheduled_by_importance[service['ru']] = [(i, service['importance'])]
        else:
            scheduled_by_importance[service['ru']].append((i, service['importance']))

    # Ordenamos la lista de tuplas de cada clave según el segundo valor (de mayor a menor)
    sorted_dict = {clave: sorted(valores, key=lambda x: x[1], reverse=True) for clave, valores in
                   scheduled_by_importance.items()}

    # Supongamos que 'sorted_dict' es el diccionario con las listas de tuplas ordenadas.
    # Ejemplo: sorted_dict = { clave: sorted(lista, key=lambda x: x[1], reverse=True) for clave, lista in dic.items() }

    # Obtenemos las claves y definimos la cantidad de subplots
    keys = list(sorted_dict.keys())
    n_keys = len(keys)
    cols = 2  # Número de columnas (puedes ajustar)
    rows = int(np.ceil(n_keys / cols))  # Número de filas

    fig = plt.figure(figsize=(17, 7))
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[0, :2])
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[0, 4:])
    ax4 = plt.subplot(gs[1, 1:3])
    ax5 = plt.subplot(gs[1, 3:5])

    # Definir los FairIndex y sus respectivos ejes
    fair_indices = ["Jain", "Gini", "Atkinson"]
    titles = ["Jain", "Gini", "Atkinson"]  # No se encontraron erratas en estos títulos.
    axes = [ax1, ax2, ax3, ax4, ax5]

    for ax, key in zip(axes, keys):
        lista = sorted_dict[key]
        # Extraer valores y definir intensidad de gris y patrones
        valores = [t[1] for t in lista]
        colores = ['0.3' if t[0] else '0.7' for t in lista]  # gris oscuro para True, gris claro para False
        # Definir el patrón de hatch (puedes ajustar los patrones)
        hatches = ['///' if t[0] else '/\\/\\' for t in lista]

        barras = ax.bar(range(1, len(lista) + 1), valores, color=colores)
        for barra, hatch in zip(barras, hatches):
            barra.set_hatch(hatch)

        ax.set_xlabel(f"Services ({len(lista)})", fontsize=12, fontweight='bold')
        ax.set_ylabel("Importance", fontsize=12, fontweight='bold')
        importance_sum = np.round(sum(t[1] for t in lista if t[0]), 4)
        ax.set_title(f"RU {key} - Sum: {importance_sum}", fontweight='bold', fontsize=14)
        ax.set_xticks([])
        #ax.set_xticklabels(range(1, len(lista) + 1), rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=10)
        # Leyenda con patrones y tonos de gris
        parche_true = mpatches.Patch(facecolor='0.3', hatch='///', label='Scheduled')
        parche_false = mpatches.Patch(facecolor='0.7', hatch='xxx', label='Not scheduled')
        ax.legend(handles=[parche_true, parche_false], fontsize=9)

    # Ajustar el diseño para evitar solapamientos (ajustamos 'top' para dejar espacio al suptitle)
    plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.85)

    # Si hay ejes de más (en caso de que la cuadrícula tenga celdas vacías), los ocultamos
    for i in range(n_keys, len(axes)):
        axes[i].set_visible(False)

    # Guardar el gráfico como PDF
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
