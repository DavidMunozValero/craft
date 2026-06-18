"""Plotting helpers for CRAFT experiments.

Visualization utilities for the analysis notebooks and the web interface.
All functions return a :class:`matplotlib.figure.Figure` so the caller
controls whether to display (``plt.show()``), save, or embed the figure
in a web page. The ``save_path`` parameter, when provided, saves the figure
to disk before returning it.
"""

from typing import Mapping, Union

import matplotlib.figure
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
) -> matplotlib.figure.Figure:
    """Box plot with strip overlay for comparing distributions across groups.

    Returns the :class:`~matplotlib.figure.Figure` so the caller can display
    or save it. When ``save_path`` is given, the figure is also saved as PDF.
    """
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

    if save_path:
        fig.savefig(
            save_path, format="pdf", dpi=300, bbox_inches="tight", transparent=True
        )
    return fig


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
) -> matplotlib.figure.Figure:
    """Line plot for convergence curves.

    Returns the :class:`~matplotlib.figure.Figure`. When ``save_path`` is
    given, the figure is also saved as PDF.
    """
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
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            frameon=True,
        )
    else:
        ax.legend(loc="lower right")

    fig.tight_layout(rect=(0, 0.15, 1, 1))

    if save_path:
        fig.savefig(
            save_path, format="pdf", dpi=300, bbox_inches="tight", transparent=True
        )
    return fig


def plot_scheduled_services(
    scheduled: np.ndarray,
    revenue_behavior: Mapping[str, Mapping[str, Union[str, float]]],
    save_path: Union[str, None] = None,
) -> matplotlib.figure.Figure:
    """Bar chart of scheduled services per RU, weighted by importance.

    Each service is a bar sized by its importance. Scheduled services use
    dark gray (``"0.3"``) with a ``"///"`` hatch; non-scheduled services use
    light gray (``"0.7"``) with a ``"/\\\\/\\\\"`` hatch. Up to 5 RU
    subplots are laid out in a 2×6 grid.

    Args:
        scheduled: Boolean array indicating which services are scheduled.
        revenue_behavior: Mapping ``service_id -> {"ru": str, "importance":
            float, ...}`` as produced by :class:`~craft.revenue.RevenueSimulator`.
        save_path: If given, save the figure as PDF before returning it.

    Returns:
        The :class:`~matplotlib.figure.Figure`.
    """
    scheduled_by_importance: dict[str, list[tuple[int, float]]] = {}
    for i, (service_id, service) in enumerate(revenue_behavior.items()):
        is_scheduled = bool(scheduled[i]) if i < len(scheduled) else False
        ru = str(service["ru"])
        importance = float(service["importance"])
        if ru not in scheduled_by_importance:
            scheduled_by_importance[ru] = [(int(is_scheduled), importance)]
        else:
            scheduled_by_importance[ru].append((int(is_scheduled), importance))

    sorted_dict = {
        clave: sorted(valores, key=lambda x: x[1], reverse=True)
        for clave, valores in scheduled_by_importance.items()
    }

    keys = list(sorted_dict.keys())
    n_keys = len(keys)

    fig = plt.figure(figsize=(17, 7))
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

    fig.subplots_adjust(hspace=0.6, wspace=0.3, top=0.85)

    for i in range(n_keys, len(axes)):
        axes[i].set_visible(False)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_convergence(
    convergence: np.ndarray,
    title: str = "Convergence",
    x_label: str = "Iteration",
    y_label: str = "Best Fitness",
    save_path: Union[str, None] = None,
    fig_size: tuple = (10, 6),
) -> matplotlib.figure.Figure:
    """Plot a single convergence curve.

    Args:
        convergence: 1-D array of per-iteration best fitness values.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        save_path: If given, save as PDF before returning.
        fig_size: Figure size.

    Returns:
        The :class:`~matplotlib.figure.Figure`.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(convergence, linewidth=2.0)
    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.grid(True, color="#A9A9A9", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=14)

    for spn in ("top", "right"):
        ax.spines[spn].set_visible(False)

    if save_path:
        fig.savefig(
            save_path, format="pdf", dpi=300, bbox_inches="tight", transparent=True
        )
    return fig
