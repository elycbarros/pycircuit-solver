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

# ---------------------------------------------------------------------------
# Waveforms for Transient Analysis
# ---------------------------------------------------------------------------

@dataclass
class StepWaveform:
    """Step waveform: v1 for t < t_step, v2 for t >= t_step."""
    v1: float = 0.0
    v2: float = 1.0
    t_step: float = 0.0

    def __call__(self, t: float) -> float:
        return self.v2 if t >= self.t_step else self.v1


@dataclass
class PulseWaveform:
    """Standard SPICE pulse waveform: PULSE(v1 v2 td tr tf pw per)."""
    v1: float
    v2: float
    td: float = 0.0       # Delay time
    tr: float = 1e-9      # Rise time
    tf: float = 1e-9      # Fall time
    pw: float = 1.0       # Pulse width
    per: float = 2.0      # Period

    def __call__(self, t: float) -> float:
        if t < self.td:
            return self.v1
        # Normalize into cycle
        t_rel = (t - self.td) % self.per if self.per > 0 else (t - self.td)
        if t_rel < self.tr:
            return self.v1 + (self.v2 - self.v1) * (t_rel / self.tr) if self.tr > 0 else self.v2
        elif t_rel < self.tr + self.pw:
            return self.v2
        elif t_rel < self.tr + self.pw + self.tf:
            t_decay = t_rel - (self.tr + self.pw)
            return self.v2 - (self.v2 - self.v1) * (t_decay / self.tf) if self.tf > 0 else self.v1
        else:
            return self.v1


@dataclass
class SineWaveform:
    """Standard SPICE sine waveform: SIN(vo va freq td theta)."""
    vo: float = 0.0      # Offset
    va: float = 1.0      # Amplitude
    freq: float = 60.0   # Frequency in Hz
    td: float = 0.0      # Delay time
    theta: float = 0.0   # Damping factor in 1/s

    def __call__(self, t: float) -> float:
        import math
        if t < self.td:
            return self.vo
        t_eff = t - self.td
        damping = math.exp(-t_eff * self.theta) if self.theta != 0 else 1.0
        return self.vo + self.va * math.sin(2.0 * math.pi * self.freq * t_eff) * damping


# ---------------------------------------------------------------------------
# Independent sources
# ---------------------------------------------------------------------------

@dataclass
class VoltageSource(_Element):
    """Independent DC / AC / Transient voltage source.

    Parameters
    ----------
    name : str
        Designator (must start with 'V').
    node_p : str
        Positive terminal node.
    node_n : str
        Negative terminal node.
    voltage : float
        DC source EMF in Volts (can be negative).
    ac_mag : float, optional
        AC peak/rms magnitude in Volts (defaults to 0.0).
    ac_phase : float, optional
        AC phase angle in degrees (defaults to 0.0).
    waveform : Callable[[float], float] | None, optional
        Time-dependent waveform function for transient analysis.
    """

    node_p: str
    node_n: str
    voltage: float
    ac_mag: float = 0.0
    ac_phase: float = 0.0
    waveform: object = None

    def value_at(self, t: float) -> float:
        """Return the instantaneous voltage at time t."""
        if callable(self.waveform):
            return float(self.waveform(t))
        return self.voltage


@dataclass
class CurrentSource(_Element):
    """Independent DC / AC / Transient current source.

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
        DC source current in Amperes (positive → current exits ``node_p``).
    ac_mag : float, optional
        AC magnitude in Amperes (defaults to 0.0).
    ac_phase : float, optional
        AC phase angle in degrees (defaults to 0.0).
    waveform : Callable[[float], float] | None, optional
        Time-dependent waveform function for transient analysis.
    """

    node_p: str
    node_n: str
    current: float
    ac_mag: float = 0.0
    ac_phase: float = 0.0
    waveform: object = None

    def value_at(self, t: float) -> float:
        """Return the instantaneous current at time t."""
        if callable(self.waveform):
            return float(self.waveform(t))
        return self.current


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


# ---------------------------------------------------------------------------
# Reactive elements (AC Analysis / Alexander & Sadiku Cap. 6, 9 & 10)
# ---------------------------------------------------------------------------

@dataclass
class Capacitor(_Element):
    """Linear ideal capacitor.

    Parameters
    ----------
    name : str
        Designator (must start with 'C').
    node_p : str
        Positive terminal node.
    node_n : str
        Negative terminal node.
    capacitance : float
        Capacitance in Farads. Must be strictly positive.
    ic : float, optional
        Initial voltage V(node_p) - V(node_n) in Volts (defaults to 0.0).
    """

    node_p: str
    node_n: str
    capacitance: float
    ic: float = 0.0

    def __post_init__(self) -> None:
        if self.capacitance <= 0:
            raise ValueError(
                f"Capacitor '{self.name}': capacitance must be > 0, got {self.capacitance}"
            )


@dataclass
class Inductor(_Element):
    """Linear ideal inductor.

    Parameters
    ----------
    name : str
        Designator (must start with 'L').
    node_p : str
        Positive terminal node.
    node_n : str
        Negative terminal node.
    inductance : float
        Inductance in Henries. Must be strictly positive.
    ic : float, optional
        Initial branch current from node_p to node_n in Amperes (defaults to 0.0).
    """

    node_p: str
    node_n: str
    inductance: float
    ic: float = 0.0

    def __post_init__(self) -> None:
        if self.inductance <= 0:
            raise ValueError(
                f"Inductor '{self.name}': inductance must be > 0, got {self.inductance}"
            )


# ---------------------------------------------------------------------------
# Current-Controlled Dependent Sources (Alexander & Sadiku Cap. 3 & 4)
# ---------------------------------------------------------------------------

@dataclass
class CCCS(_Element):
    """Current-Controlled Current Source (F element in SPICE).

    ``I_out = gain * I(vctrl_name)``

    Current flows from ``node_p`` to ``node_n`` through the external circuit.

    Parameters
    ----------
    name : str
        Designator (must start with 'F').
    node_p : str
        Node where controlled current exits.
    node_n : str
        Node where controlled current enters.
    vctrl_name : str
        Name of the independent voltage source whose branch current controls this source.
    gain : float
        Dimensionless current gain.
    """

    node_p: str
    node_n: str
    vctrl_name: str
    gain: float


@dataclass
class CCVS(_Element):
    """Current-Controlled Voltage Source (H element in SPICE).

    ``V_out = transresistance * I(vctrl_name)``

    Parameters
    ----------
    name : str
        Designator (must start with 'H').
    node_p : str
        Positive output terminal.
    node_n : str
        Negative output terminal.
    vctrl_name : str
        Name of the independent voltage source whose branch current controls this source.
    transresistance : float
        Transresistance in Ohms (V/A).
    """

    node_p: str
    node_n: str
    vctrl_name: str
    transresistance: float

