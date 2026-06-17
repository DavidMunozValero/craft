"""mealpy-based optimization backend.

Formulation of the revenue-maximization (and fairness-oriented) railway
timetabling problem to be solved with the algorithms provided by the
`mealpy <https://github.com/thieu199h/mealpy>`_ library.
"""

from .timetabling import MealpyTimetabling

__all__ = ["MealpyTimetabling"]
