"""
circuit_solver — MNA-based linear circuit analysis engine (DC & AC steady-state).

Public re-exports for convenient top-level access.
"""

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    CCCS,
    CCVS,
    OpAmp,
)
from circuit_solver.core.solution import Solution
from circuit_solver.engine.mna import solve
from circuit_solver.engine.ac import solve_ac, ACSolution, ac_sweep
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.engine.monte_carlo import monte_carlo
from circuit_solver.visualization.plots import (
    plot_power_transfer,
    plot_monte_carlo,
    plot_monte_carlo_all,
    plot_bode,
)

__all__ = [
    "Circuit",
    "Resistor",
    "Capacitor",
    "Inductor",
    "VoltageSource",
    "CurrentSource",
    "VCVS",
    "VCCS",
    "CCCS",
    "CCVS",
    "OpAmp",
    "Solution",
    "ACSolution",
    "solve",
    "solve_ac",
    "ac_sweep",
    "thevenin_norton",
    "monte_carlo",
    "plot_power_transfer",
    "plot_monte_carlo",
    "plot_monte_carlo_all",
    "plot_bode",
]
