"""Visual analytics for the direct genetic encoding.

Plot helpers to review individuals and schedules while the GA evolves:

* :func:`plot_genome` renders one individual as a heatmap (services x genes),
  with blue for advances, red for delays and gray for dropped services.
* :func:`plot_genome_filmstrip` renders several individuals side by side with
  a shared color scale, so the evolution of the best genome is visible at a
  glance.
* :func:`plot_timetable` renders one individual as a Marey (space-time)
  diagram over the requested timetables, highlighting the shifted services.
* :func:`plot_timetable_filmstrip` renders timetable snapshots with service
  colors kept consistent across panels.
* :func:`plot_evolution` summarizes a run history (fitness, planned services
  and conflicts per generation).

Every service keeps the same color in every view and panel: the evolution
can be tracked service by service.
"""

from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, to_hex
from matplotlib.lines import Line2D

from .base import DiagnosableProblem

DIVERGING = "RdBu_r"
UNPLANNED_COLOR = "0.93"
REQUESTED_COLOR = "0.75"
SERVICE_CMAP = "tab20"


def _chronological_order(problem: DiagnosableProblem) -> List[int]:
    """Service indices sorted by requested origin departure."""

    def origin_departure(i: int) -> float:
        schedule = problem.requested_schedules[problem.service_ids[i]]
        return float(schedule[list(schedule)[0]][1])

    return sorted(range(problem.n_services), key=origin_departure)


def _delta_matrix(
    problem: DiagnosableProblem, genome: np.ndarray, order: Sequence[int]
) -> np.ma.MaskedArray:
    """Deltas of a genome as a masked matrix (services x delta genes).

    Each planned service places its deltas left-aligned on its own row;
    dropped services keep the whole row masked so it renders as background.
    """
    width = max(
        problem.slices[i].stop - problem.slices[i].start - 1
        for i in range(problem.n_services)
    )
    matrix = np.full((problem.n_services, width), np.nan)
    for row, i in enumerate(order):
        if not genome[problem.slices[i].start]:
            continue
        deltas = genome[problem.time_slices[i]]
        matrix[row, : len(deltas)] = deltas
    masked: np.ma.MaskedArray = np.ma.masked_invalid(matrix)
    return masked


def _genome_norm(
    problem: DiagnosableProblem, genomes: Sequence[np.ndarray]
) -> TwoSlopeNorm:
    """Shared diverging norm centered on zero for the given genomes."""
    values = [
        abs(gene)
        for genome in genomes
        for i in range(problem.n_services)
        if genome[problem.slices[i].start]
        for gene in genome[problem.time_slices[i]]
    ]
    cap = max(max(values, default=0), 1)
    return TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap)


def service_colors(problem: DiagnosableProblem) -> Dict[str, str]:
    """Stable color per service, shared by every plotting helper."""
    palette = plt.get_cmap(SERVICE_CMAP)
    return {
        sid: to_hex(palette(((i % palette.N) + 0.5) / palette.N))
        for i, sid in enumerate(problem.service_ids)
    }


def subset_evenly_by_departure(problem: DiagnosableProblem, count: int) -> List[str]:
    """Pick ``count`` services spread over the day by requested departure."""
    order = _chronological_order(problem)
    step = max(1, problem.n_services // count)
    picked = [problem.service_ids[i] for i in order[::step]][:count]
    return picked


def plot_genome(
    problem: DiagnosableProblem,
    genome: np.ndarray,
    ax: Optional[plt.Axes] = None,
    norm: Optional[TwoSlopeNorm] = None,
    show_labels: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """Render a genome as a heatmap: one row per service, one column per gene.

    Rows are ordered chronologically by requested origin departure. Dropped
    services appear as gray rows (their deltas are not decoded); planned
    services show their deltas with a diverging scale (blue = advance, red =
    delay) bounded by ``norm``.

    Args:
        problem: DiagnosableProblem defining the genome layout.
        genome: Flat integer genome.
        ax: Axes to draw on (created on demand when omitted).
        norm: Shared color norm (computed from ``genome`` when omitted).
        show_labels: Whether to annotate service ids on the y axis.
        title: Optional axes title.

    Returns:
        The axes holding the heatmap (with an attached colorbar when the
        axes were created on demand).
    """
    standalone = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.22 * problem.n_services + 1.5))
    if norm is None:
        norm = _genome_norm(problem, [genome])

    order = _chronological_order(problem)
    planned = [bool(genome[problem.slices[i].start]) for i in order]

    cmap = plt.get_cmap(DIVERGING).with_extremes(bad=UNPLANNED_COLOR)
    image = ax.imshow(
        _delta_matrix(problem, genome, order),
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    if show_labels:
        labels = [
            f"{'' if flag else '✗ '}{problem.service_ids[i]}"
            for i, flag in zip(order, planned)
        ]
        ax.set_yticks(range(problem.n_services))
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=9)
    if standalone:
        ax.figure.colorbar(image, ax=ax, label="δ (min)")
    return ax


def plot_genome_filmstrip(
    problem: DiagnosableProblem,
    genomes: Sequence[np.ndarray],
    titles: Sequence[str],
    ncols: int = 2,
) -> plt.Figure:
    """Render genome snapshots side by side with a shared color scale.

    Args:
        problem: DiagnosableProblem defining the genome layout.
        genomes: Genome snapshots (e.g. ``ga.best_genomes[steps]``).
        titles: One title per snapshot.
        ncols: Columns of the grid.

    Returns:
        The figure holding the filmstrip.
    """
    genomes = list(genomes)
    nrows = -(-len(genomes) // ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 0.14 * problem.n_services * nrows + 1.6),
        squeeze=False,
        sharey=True,
    )
    flat = axes.ravel()
    norm = _genome_norm(problem, genomes)
    for k, (genome, title) in enumerate(zip(genomes, titles)):
        plot_genome(problem, genome, ax=flat[k], norm=norm, title=title)
    for ax in flat[len(genomes) :]:
        ax.axis("off")

    fig.colorbar(flat[0].images[0], ax=list(flat[: len(genomes)]), label="δ (min)")
    return fig


def _marey(
    ax: plt.Axes,
    schedules: Mapping[str, Mapping[str, Sequence[float]]],
    station_km: Mapping[str, float],
    color: str,
    ls: str,
    lw: float,
    alpha: float,
) -> None:
    """Draw schedules as space-time lines on ``ax``."""
    for schedule in schedules.values():
        xs: List[float] = []
        ys: List[float] = []
        for station, (arrival, departure) in schedule.items():
            xs.extend([arrival, departure])
            ys.extend([station_km[station]] * 2)
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, alpha=alpha)


def plot_timetable(
    problem: DiagnosableProblem,
    genome: np.ndarray,
    ax: Optional[plt.Axes] = None,
    subset: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Render a genome as a Marey (space-time) diagram.

    The requested timetables of ``subset`` (all services by default) are
    drawn in light gray; every planned service of the genome is drawn with
    its decoded schedule in its own color, and when its times were shifted
    the requested timetable is redrawn dashed in the same color, so the
    carried-over shift is directly visible.

    Args:
        problem: DiagnosableProblem defining the genome layout (single-corridor).
        genome: Flat integer genome.
        ax: Axes to draw on (created on demand when omitted).
        subset: Service ids to draw (all by default).
        title: Optional axes title.

    Returns:
        The axes holding the diagram.
    """
    if len(problem.conflict_checkers) != 1:
        raise ValueError("timetable plots support single-corridor instances")
    checker = problem.conflict_checkers[0]
    stations = {sta.id: sta for svc in problem.services for sta in svc.line.stations}
    colors = service_colors(problem)

    requested = problem.requested_schedules
    decoded = problem.decoded_schedule(genome)
    planned_ids = set(_planned_ids(problem, genome))
    drawn = set(subset) if subset is not None else set(problem.service_ids)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7))
    background = {sid: requested[sid] for sid in drawn if sid not in planned_ids}
    _marey(ax, background, checker.km, REQUESTED_COLOR, "-", 0.7, 0.55)
    for sid in drawn & planned_ids:
        i = problem.row_index[sid]
        shifted = bool(np.any(genome[problem.time_slices[i]]))
        if shifted:
            _marey(ax, {sid: requested[sid]}, checker.km, colors[sid], "--", 1.0, 0.9)
        _marey(ax, {sid: decoded[sid]}, checker.km, colors[sid], "-", 1.8, 0.95)

    ax.set_xticks(list(range(5 * 60, 24 * 60, 120)))
    ax.set_xticklabels(
        [f"{t // 60:02d}:{t % 60:02d}" for t in ax.get_xticks()],
        rotation=45,
        fontsize=8,
    )
    ax.set_yticks([checker.km[s] for s in checker.corridor_ids])
    ax.set_yticklabels(
        [stations[s].short_name if s in stations else s for s in checker.corridor_ids],
        fontsize=8,
    )
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def _planned_ids(problem: DiagnosableProblem, genome: np.ndarray) -> List[str]:
    """Service ids planned by a genome."""
    return [
        problem.service_ids[i]
        for i in range(problem.n_services)
        if genome[problem.slices[i].start]
    ]


def plot_timetable_filmstrip(
    problem: DiagnosableProblem,
    genomes: Sequence[np.ndarray],
    titles: Sequence[str],
    subset: Optional[Sequence[str]] = None,
    ncols: int = 2,
) -> plt.Figure:
    """Render timetable snapshots with service colors consistent across panels.

    Args:
        problem: DiagnosableProblem defining the genome layout (single-corridor).
        genomes: Genome snapshots (e.g. ``ga.best_genomes[steps]``).
        titles: One title per snapshot.
        subset: Service ids to draw (all by default).
        ncols: Columns of the grid.

    Returns:
        The figure holding the filmstrip.
    """
    genomes = list(genomes)
    nrows = -(-len(genomes) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.4 * ncols, 4.6 * nrows), squeeze=False
    )
    flat = axes.ravel()
    for k, (genome, title) in enumerate(zip(genomes, titles)):
        plot_timetable(problem, genome, ax=flat[k], subset=subset, title=title)

    for ax in flat[len(genomes) :]:
        ax.axis("off")
    handles = [
        Line2D(
            [0], [0], color=REQUESTED_COLOR, lw=1.0, label="solicitado no planificado"
        ),
        Line2D([0], [0], color="0.35", lw=1.6, label="planificado sin cambio"),
        Line2D(
            [0], [0], color="0.35", lw=1.2, ls="--", label="solicitado del desviado"
        ),
    ]
    flat[0].legend(handles=handles, loc="upper left", fontsize=7)
    return fig


def plot_evolution(history: pd.DataFrame) -> plt.Figure:
    """Summarize a run history: fitness, planned services and conflicts.

    Args:
        history: The DataFrame returned by :meth:`craft.ga.algorithm.GA.optimize`.

    Returns:
        The figure with the convergence curve and the best-individual
        evolution (planned services and conflicting pairs on separate axes).
    """
    fig, (fitness_ax, counts_ax) = plt.subplots(1, 2, figsize=(14, 4.5))

    fitness_ax.plot(history["Iteration"], history["Fitness"], color="C0", lw=1.8)
    fitness_ax.set_xlabel("Generación")
    fitness_ax.set_ylabel("Fitness (revenue − penalización)")
    fitness_ax.set_title("Convergencia")
    fitness_ax.grid(alpha=0.3)

    counts_ax.step(
        history["Iteration"],
        history["Planned"],
        where="post",
        color="C2",
        label="servicios planificados",
    )
    conflicts_ax = counts_ax.twinx()
    conflicts_ax.step(
        history["Iteration"],
        history["Conflicts"],
        where="post",
        color="C3",
        label="pares en conflicto",
    )
    counts_ax.set_xlabel("Generación")
    counts_ax.set_ylabel("Servicios planificados", color="C2")
    conflicts_ax.set_ylabel("Pares en conflicto", color="C3")
    counts_ax.set_title("Mejor individuo por generación")
    counts_ax.grid(alpha=0.3)

    handles = [
        Line2D([0], [0], color="C2", lw=1.8, label="planificados (eje izq.)"),
        Line2D([0], [0], color="C3", lw=1.8, label="conflictos (eje der.)"),
    ]
    counts_ax.legend(handles=handles, loc="center right", fontsize=8)
    fig.tight_layout()
    return fig
