"""
elements.py — Dataclass definitions for every supported circuit element.

Supported elements
------------------
* Resistor   — passive linear resistor
* VoltageSource — independent DC voltage source
* CurrentSource — independent DC current source
* VCVS       — Voltage-Controlled Voltage Source  (E element, SPICE)
* VCCS       — Voltage-Controlled Current Source  (G element, SPICE)
* OpAmp      — Ideal operational amplifier (infinite gain, infinite Rin)

All numeric values are stored in SI base units (Ω, V, A) after conversion
from engineering-suffix notation by :mod:`circuit_solver.core.validator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class _Element:
    """Abstract base for all two-terminal and multi-terminal elements."""
    name: str  # SPICE-style designator, e.g. "R1", "V2", "E3"


# ---------------------------------------------------------------------------
# Passive
# ---------------------------------------------------------------------------

@dataclass
class Resistor(_Element):
    """Linear resistor.

    Parameters
    ----------
    name : str
        Designator (must start with 'R').
    node_p : str
        Positive terminal node label.
    node_n : str
        Negative terminal node label (often '0' for ground).
    resistance : float
        Resistance in Ohms. Must be strictly positive.
    tolerance : float, optional
        Fractional tolerance for Monte Carlo analysis (e.g. 0.05 for ±5 %).
        Defaults to 0.05.
    """

    node_p: str
    node_n: str
    resistance: float
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.resistance <= 0:
            raise ValueError(
                f"Resistor '{self.name}': resistance must be > 0, got {self.resistance}"
            )


# ---------------------------------------------------------------------------
# Independent sources
# ---------------------------------------------------------------------------

@dataclass
class VoltageSource(_Element):
    """Independent DC voltage source.

    Parameters
    ----------
    name : str
        Designator (must start with 'V').
    node_p : str
        Positive terminal node.
    node_n : str
        Negative terminal node.
    voltage : float
        Source EMF in Volts (can be negative).
    """

    node_p: str
    node_n: str
    voltage: float


@dataclass
class CurrentSource(_Element):
    """Independent DC current source.

    Conventional current direction: *from* ``node_n`` *into* ``node_p``
    (i.e. current enters the positive terminal of the source).

    Parameters
    ----------
    name : str
        Designator (must start with 'I').
    node_p : str
        Node where current *exits* the source.
    node_n : str
        Node where current *enters* the source.
    current : float
        Source current in Amperes (positive → current exits ``node_p``).
    """

    node_p: str
    node_n: str
    current: float


# ---------------------------------------------------------------------------
# Dependent sources
# ---------------------------------------------------------------------------

@dataclass
class VCVS(_Element):
    """Voltage-Controlled Voltage Source (E element in SPICE).

    ``V_out = gain * (V_ctrl_p − V_ctrl_n)``

    Parameters
    ----------
    name : str
        Designator (must start with 'E').
    node_p : str
        Positive output terminal.
    node_n : str
        Negative output terminal.
    ctrl_p : str
        Positive controlling node.
    ctrl_n : str
        Negative controlling node.
    gain : float
        Dimensionless voltage gain.
    """

    node_p: str
    node_n: str
    ctrl_p: str
    ctrl_n: str
    gain: float


@dataclass
class VCCS(_Element):
    """Voltage-Controlled Current Source (G element in SPICE).

    ``I_out = transconductance * (V_ctrl_p − V_ctrl_n)``

    Current flows from ``node_p`` to ``node_n`` through the external circuit.

    Parameters
    ----------
    name : str
        Designator (must start with 'G').
    node_p : str
        Node where controlled current *exits*.
    node_n : str
        Node where controlled current *enters*.
    ctrl_p : str
        Positive controlling node.
    ctrl_n : str
        Negative controlling node.
    transconductance : float
        Transconductance in Siemens (A/V).
    """

    node_p: str
    node_n: str
    ctrl_p: str
    ctrl_n: str
    transconductance: float


# ---------------------------------------------------------------------------
# Op-Amp
# ---------------------------------------------------------------------------

@dataclass
class OpAmp(_Element):
    """Ideal operational amplifier (infinite open-loop gain).

    The MNA model enforces the virtual-short constraint:
    ``V(in_p) = V(in_n)`` and introduces an auxiliary current variable
    for the output branch.

    Parameters
    ----------
    name : str
        Designator (must start with 'A' or 'U').
    in_p : str
        Non-inverting input node (+).
    in_n : str
        Inverting input node (−).
    out : str
        Output node.
    gnd : str
        Power-supply reference node (typically '0').
        The model assumes ideal rails; ``gnd`` participates only in the
        output KCL stamp, not in the virtual-short equation.
    """

    in_p: str
    in_n: str
    out: str
    gnd: str = "0"
