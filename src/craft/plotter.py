"""Plotting helpers for CRAFT experiments.

Visualization utilities used by the analysis notebooks: box/line plots for
comparing optimization runs and a scheduled-services chart that summarizes
which services of each RU are scheduled weighted by their importance.
"""

from typing import Mapping, Union

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sns_box_plot(
    df: pd.DataFrame,
    x_data: str,
    y_data: str,
    title: str,
    x_label: str,
    y_label: str,
    hue: Union[str, None] = None,
    save_path: Union[str, None] = None,
    fig_size: tuple = (10, 6),
) -> None:
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_title(title, fontweight="bold", fontsize=18)

    sns.boxplot(
        data=df,
        x=x_data,
        y=y_data,
        hue=hue,
        dodge=True,
        zorder=1,
        boxprops=dict(alpha=0.3),
        ax=ax,
    )
    sns.stripplot(
        data=df, x=x_data, y=y_data, hue=hue, dodge=True, alpha=0.5, zorder=1, ax=ax
    )

    handles, labels = ax.get_legend_handles_labels()
    new_handles = [
        handle
        for handle, label in zip(handles, labels)
        if "line" not in str(type(handle))
    ]

    if hue:
        ax.legend(handles=new_handles, title=hue, fontsize=12, title_fontsize=14)

    ax.grid(axis="y", color="#A9A9A9", alpha=0.3, zorder=1)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    for spn in ("top", "right", "bottom", "left"):
        ax.spines[spn].set_visible(True)
        ax.spines[spn].set_linewidth(1.0)
        ax.spines[spn].set_color("#A9A9A9")

    plt.show()
    if save_path:
        fig.savefig(
            save_path, format="pdf", dpi=300, bbox_inches="tight", transparent=True
        )


def sns_line_plot(
    df: pd.DataFrame,
    x_data: str,
    y_data: str,
    title: str,
    x_label: str,
    y_label: str,
    hue: Union[str, None] = None,
    save_path: Union[str, None] = None,
    legend_type: str = "outside",
    x_limit: tuple = (-1, 100),
    y_limit: tuple = (-1, 4000),
    fig_size: tuple = (10, 6),
) -> None:
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    sns.lineplot(ax=ax, data=df, x=x_data, y=y_data, hue=hue, legend=True)

    ax.grid(axis="y", color="#A9A9A9", alpha=0.3, zorder=1)

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    for spn in ("top", "right", "bottom", "left"):
        ax.spines[spn].set_visible(True)
        ax.spines[spn].set_linewidth(1.0)
        ax.spines[spn].set_color("#A9A9A9")

    if legend_type == "outside":
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            frameon=True,
        )
    else:
        plt.legend(loc="lower right")

    plt.tight_layout(rect=(0, 0.15, 1, 1))

    plt.show()
    if save_path:
        fig.savefig(
            save_path, format="pdf", dpi=300, bbox_inches="tight", transparent=True
        )


def plot_scheduled_services(
    fair_index: str,
    summary_df: pd.DataFrame,
    df_history: pd.DataFrame,
    revenue_behavior: Mapping[str, Mapping[str, Union[str, float]]],
    save_path: str,
) -> None:
    """Plot scheduled services per RU, weighted by importance.

    For the run with minimum inequity under the requested ``fair_index``,
    show a bar chart per RU where each service is a bar sized by its
    importance and hatched/colored depending on whether it is scheduled.
    """
    idx_min = summary_df.groupby("FairIndex")["Inequity"].idxmin()
    resultado = summary_df.loc[idx_min, ["FairIndex", "Run"]]

    df_history_unique = df_history.groupby(["FairIndex", "Run"], as_index=False)[
        "Discrete"
    ].first()
    resultado_final = resultado.merge(
        df_history_unique[["FairIndex", "Run", "Discrete"]],
        on=["FairIndex", "Run"],
        how="left",
    )

    scheduled_services = None
    for row in resultado_final.iterrows():
        if row[1]["FairIndex"] == fair_index:
            scheduled_services = row[1]["Discrete"]

    scheduled_by_importance: dict[str, list[tuple[int, float]]] = {}
    for i, t in zip(scheduled_services or [], revenue_behavior.items()):
        service_id, service = t
        ru = str(service["ru"])
        importance = float(service["importance"])
        if ru not in scheduled_by_importance:
            scheduled_by_importance[ru] = [(i, importance)]
        else:
            scheduled_by_importance[ru].append((i, importance))

    sorted_dict = {
        clave: sorted(valores, key=lambda x: x[1], reverse=True)
        for clave, valores in scheduled_by_importance.items()
    }

    keys = list(sorted_dict.keys())
    n_keys = len(keys)

    plt.figure(figsize=(17, 7))
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[0, :2])
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[0, 4:])
    ax4 = plt.subplot(gs[1, 1:3])
    ax5 = plt.subplot(gs[1, 3:5])

    axes = [ax1, ax2, ax3, ax4, ax5]

    for ax, key in zip(axes, keys):
        lista = sorted_dict[key]
        valores = [t[1] for t in lista]
        colores = ["0.3" if t[0] else "0.7" for t in lista]
        hatches = ["///" if t[0] else "/\\/\\" for t in lista]

        barras = ax.bar(range(1, len(lista) + 1), valores, color=colores)
        for barra, hatch in zip(barras, hatches):
            barra.set_hatch(hatch)

        ax.set_xlabel(f"Services ({len(lista)})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Importance", fontsize=12, fontweight="bold")
        importance_sum = np.round(sum(t[1] for t in lista if bool(t[0])), 4)
        ax.set_title(
            f"RU {key} - Sum: {importance_sum}", fontweight="bold", fontsize=14
        )
        ax.set_xticks([])
        ax.tick_params(axis="both", which="major", labelsize=10)
        parche_true = mpatches.Patch(facecolor="0.3", hatch="///", label="Scheduled")
        parche_false = mpatches.Patch(
            facecolor="0.7", hatch="xxx", label="Not scheduled"
        )
        ax.legend(handles=[parche_true, parche_false], fontsize=9)

    plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.85)

    for i in range(n_keys, len(axes)):
        axes[i].set_visible(False)

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
