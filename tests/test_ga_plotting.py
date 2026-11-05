"""Tests for the GA visual analytics (``craft.ga.plotting``)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from craft.ga import GA, GATimetabling  # noqa: E402
from craft.ga.plotting import (  # noqa: E402
    plot_evolution,
    plot_genome,
    plot_genome_filmstrip,
    plot_timetable,
    plot_timetable_filmstrip,
    subset_evenly_by_departure,
)


@pytest.fixture()
def problem(supply, revenue_behavior) -> GATimetabling:
    """Timetabling problem over the session fixture supply."""
    return GATimetabling(supply, revenue_behavior)


@pytest.fixture()
def snapshots(problem):
    """Best genomes of a short run, plus their generation labels."""
    ga = GA(problem, pop_size=12)
    history = ga.optimize(epoch=4, seed=0, verbose=False)
    steps = [0, 2, 4]
    return ga.best_genomes[steps], [f"generación {s}" for s in steps], history


def test_plot_genome_renders_every_service_row(problem):
    """The standalone heatmap carries one row per service and a colorbar."""
    genome = np.zeros(problem.genome_size, dtype=np.int64)
    for block in problem.slices:
        genome[block.start] = 1
    ax = plot_genome(problem, genome)
    assert len(ax.images) == 1
    assert len(ax.figure.axes) == 2
    plt.close(ax.figure)


def test_genome_filmstrip_shares_the_scale(problem, snapshots):
    """The filmstrip builds one panel per snapshot with a shared colorbar."""
    genomes, titles, _ = snapshots
    fig = plot_genome_filmstrip(problem, genomes, titles)
    assert len(fig.axes) == 5  # 2x2 grid (one panel left off) + shared colorbar
    plt.close(fig)


def test_timetable_filmstrip_draws_snapshots(problem, snapshots):
    """The timetable filmstrip draws one Marey panel per snapshot."""
    genomes, titles, _ = snapshots
    fig = plot_timetable_filmstrip(problem, genomes, titles)
    assert len(fig.axes) == 4  # 2x2 grid (one panel left off)
    plt.close(fig)


def test_plot_timetable_draws_requested_and_planned(problem):
    """The single Marey view draws one line per planned service plus the
    requested line of the shifted ones."""
    subset = subset_evenly_by_departure(problem, 4)
    genome = np.zeros(problem.genome_size, dtype=np.int64)
    for block in problem.slices:
        genome[block.start] = 1
    shifted = subset[0]
    genome[problem.time_slices[problem.row_index[shifted]].start] = 5

    ax = plot_timetable(problem, genome, subset=subset)
    assert len(ax.lines) == len(subset) + 1
    plt.close(ax.figure)


def test_plot_evolution_summarizes_the_run(problem, snapshots):
    """The evolution figure exposes the convergence and count panels."""
    _, _, history = snapshots
    fig = plot_evolution(history)
    assert len(fig.axes) == 3  # fitness panel + counts panel + twin axis
    plt.close(fig)
