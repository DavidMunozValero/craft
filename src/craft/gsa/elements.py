"""Dynamic elements of the Gravitational Search Algorithm (GSA).

Lightweight containers for the per-agent dynamic quantities manipulated
during the GSA iterations: velocity, acceleration and the gravitational
constant, each split into real and discrete parts to match the hybrid
solution representation.
"""

from typing import Any, List, Tuple, Union


class Velocity:
    """Velocity of an agent, split into real and discrete parts."""

    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


class Acceleration:
    """Acceleration of an agent, split into real and discrete parts."""

    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


class GConstant:
    """Gravitational constant, split into real and discrete parts."""

    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete
