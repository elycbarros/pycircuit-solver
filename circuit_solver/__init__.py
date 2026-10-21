"""
circuit_solver — MNA-based linear circuit analysis engine (DC & AC steady-state).

Public re-exports for convenient top-level access.
"""

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    Capacitor,
    Inductor,
    MutualCoupling,
    Transformer,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    CCCS,
    CCVS,
    OpAmp,
    Diode,
    ZenerDiode,
    BJT,
    MOSFET,
    StepWaveform,
    PulseWaveform,
    SineWaveform,
)
from circuit_solver.core.solution import Solution
from circuit_solver.core.transient_solution import TransientSolution
from circuit_solver.engine.mna import solve
from circuit_solver.engine.nonlinear import solve_nonlinear
from circuit_solver.engine.ac import solve_ac, ACSolution, ac_sweep
from circuit_solver.engine.transient import solve_transient
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.engine.monte_carlo import monte_carlo
from circuit_solver.engine.dc_sweep import dc_sweep, DCSweepResult
from circuit_solver.visualization.plots import (
    plot_power_transfer,
    plot_monte_carlo,
    plot_monte_carlo_all,
    plot_bode,
    plot_transient,
    plot_dc_sweep,
)

__all__ = [
    "Circuit",
    "Resistor",
    "Capacitor",
    "Inductor",
    "MutualCoupling",
    "Transformer",
    "VoltageSource",
    "CurrentSource",
    "VCVS",
    "VCCS",
    "CCCS",
    "CCVS",
    "OpAmp",
    "Diode",
    "ZenerDiode",
    "BJT",
    "MOSFET",
    "StepWaveform",
    "PulseWaveform",
    "SineWaveform",
    "Solution",
    "ACSolution",
    "TransientSolution",
    "DCSweepResult",
    "solve",
    "solve_nonlinear",
    "solve_ac",
    "ac_sweep",
    "solve_transient",
    "thevenin_norton",
    "monte_carlo",
    "dc_sweep",
    "plot_power_transfer",
    "plot_monte_carlo",
    "plot_monte_carlo_all",
    "plot_bode",
    "plot_transient",
    "plot_dc_sweep",
]
