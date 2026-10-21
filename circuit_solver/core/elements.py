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


@dataclass
class MutualCoupling(_Element):
    """Magnetic mutual coupling between two inductors (K element in SPICE).

    Constitutive equation:
        M = coupling * sqrt(L1 * L2)

    Parameters
    ----------
    name : str
        Designator (must start with 'K').
    ind1_name : str
        Designator of first inductor (e.g. 'L1').
    ind2_name : str
        Designator of second inductor (e.g. 'L2').
    coupling : float
        Magnetic coupling coefficient k (-1.0 <= k <= 1.0, default 0.999).
    """

    ind1_name: str
    ind2_name: str
    coupling: float = 0.999

    def __post_init__(self) -> None:
        if abs(self.coupling) > 1.0:
            raise ValueError(
                f"Coupling coefficient k must satisfy -1.0 <= k <= 1.0 (got {self.coupling})."
            )

    def mutual_inductance(self, L1: float, L2: float) -> float:
        """Mutual inductance M = k * sqrt(L1 * L2)."""
        import math
        return self.coupling * math.sqrt(L1 * L2)


@dataclass
class Transformer(_Element):
    """Two-winding coupled transformer.

    Parameters
    ----------
    name : str
        Designator (must start with 'T' or 'K').
    p1 : str
        Primary positive terminal (dot).
    n1 : str
        Primary negative terminal.
    p2 : str
        Secondary positive terminal (dot).
    n2 : str
        Secondary negative terminal.
    L1 : float
        Primary self-inductance in Henries.
    L2 : float
        Secondary self-inductance in Henries.
    coupling : float, optional
        Coupling coefficient k (-1.0 <= k <= 1.0, default 0.999).
    """

    p1: str
    n1: str
    p2: str
    n2: str
    L1: float
    L2: float
    coupling: float = 0.999

    def __post_init__(self) -> None:
        if self.L1 <= 0 or self.L2 <= 0:
            raise ValueError(f"Transformer '{self.name}': L1 and L2 must be > 0.")
        if abs(self.coupling) > 1.0:
            raise ValueError(
                f"Coupling coefficient k must satisfy -1.0 <= k <= 1.0 (got {self.coupling})."
            )

    @property
    def turns_ratio(self) -> float:
        """Ideal turns ratio n = N1 / N2 = sqrt(L1 / L2)."""
        import math
        return math.sqrt(self.L1 / self.L2)

    @property
    def mutual_inductance(self) -> float:
        """Mutual inductance M = k * sqrt(L1 * L2)."""
        import math
        return self.coupling * math.sqrt(self.L1 * self.L2)


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


# ---------------------------------------------------------------------------
# Non-linear Semiconductor Elements (Diodes & Transistors)
# ---------------------------------------------------------------------------

@dataclass
class Diode(_Element):
    """PN Junction Diode (Shockley model).

    Current equation:
        I_d = Is * (exp(V_d / (n * Vt)) - 1)

    Dynamic conductance:
        g_d = d(I_d)/d(V_d) = (Is / (n * Vt)) * exp(V_d / (n * Vt))

    Parameters
    ----------
    name : str
        Designator (must start with 'D').
    node_p : str
        Anode terminal node (+).
    node_n : str
        Cathode terminal node (-).
    Is : float, optional
        Saturation current in Amperes (default 1e-14 A).
    n : float, optional
        Emission coefficient / ideality factor (default 1.0).
    Vt : float, optional
        Thermal voltage in Volts (default 0.02585 V ≈ 26 mV at 300 K).
    """

    node_p: str
    node_n: str
    Is: float = 1e-14
    n: float = 1.0
    Vt: float = 0.02585

    def evaluate(self, v_d: float) -> tuple[float, float]:
        """Evaluate diode current I_d and dynamic conductance g_d at voltage v_d.

        Applies SPICE-style exponential voltage clamping to prevent numeric overflow.
        """
        import math
        # Clamp v_d to avoid math range error
        v_crit = self.n * self.Vt * math.log((self.n * self.Vt) / (math.sqrt(2.0) * self.Is))
        v_clamped = min(v_d, 0.85)

        arg = v_clamped / (self.n * self.Vt)
        if arg > 80.0:
            arg = 80.0
        elif arg < -80.0:
            # Fully reverse-biased
            return -self.Is, 1e-12

        exp_val = math.exp(arg)
        i_d = self.Is * (exp_val - 1.0)
        g_d = (self.Is / (self.n * self.Vt)) * exp_val
        g_d = max(g_d, 1e-12)  # Minimum numerical conductance

        # If clamped, use linear extrapolation beyond clamp threshold
        if v_d > v_clamped:
            i_d += g_d * (v_d - v_clamped)

        return i_d, g_d


@dataclass
class ZenerDiode(_Element):
    """Zener Diode with reverse breakdown regulation.

    Forward: standard PN diode.
    Reverse breakdown: begins conducting sharply when V_d <= -Vz with slope 1/Rz.

    Parameters
    ----------
    name : str
        Designator (must start with 'D' or 'Z').
    node_p : str
        Anode node.
    node_n : str
        Cathode node.
    Vz : float
        Zener breakdown voltage in Volts (positive value, e.g. 5.1 V).
    Rz : float, optional
        Zener dynamic resistance in breakdown (default 10.0 Ω).
    Is : float, optional
        Forward saturation current (default 1e-14 A).
    n : float, optional
        Forward ideality factor (default 1.0).
    Vt : float, optional
        Thermal voltage (default 0.02585 V).
    """

    node_p: str
    node_n: str
    Vz: float
    Rz: float = 10.0
    Is: float = 1e-14
    n: float = 1.0
    Vt: float = 0.02585

    def evaluate(self, v_d: float) -> tuple[float, float]:
        """Evaluate Zener current and conductance at voltage v_d = V(p) - V(n)."""
        import math
        if v_d >= -self.Vz:
            # Normal forward / off region (same as Diode)
            v_clamped = min(v_d, 0.85)
            arg = max(min(v_clamped / (self.n * self.Vt), 80.0), -80.0)
            exp_val = math.exp(arg)
            i_d = self.Is * (exp_val - 1.0)
            g_d = max((self.Is / (self.n * self.Vt)) * exp_val, 1e-12)
            if v_d > v_clamped:
                i_d += g_d * (v_d - v_clamped)
            return i_d, g_d
        else:
            # Reverse breakdown region (Zener effect)
            delta_v = (-self.Vz) - v_d  # positive when deep in breakdown
            gz = 1.0 / max(self.Rz, 1e-3)
            i_d = -self.Is - gz * delta_v
            g_d = gz
            return i_d, g_d


@dataclass
class BJT(_Element):
    """Bipolar Junction Transistor (Ebers-Moll / active-saturation model).

    Parameters
    ----------
    name : str
        Designator (must start with 'Q').
    node_c : str
        Collector node.
    node_b : str
        Base node.
    node_e : str
        Emitter node.
    polarity : {"NPN", "PNP"}, optional
        Transistor polarity (default "NPN").
    beta : float, optional
        Forward common-emitter current gain h_FE (default 100.0).
    Is : float, optional
        Base-emitter saturation current (default 1e-14 A).
    Vt : float, optional
        Thermal voltage (default 0.02585 V).
    """

    node_c: str
    node_b: str
    node_e: str
    polarity: Literal["NPN", "PNP"] = "NPN"
    beta: float = 100.0
    Is: float = 1e-14
    Vt: float = 0.02585

    def evaluate(
        self, v_c: float, v_b: float, v_e: float
    ) -> tuple[
        dict[str, float],              # currents: {"C": ic, "B": ib, "E": ie}
        dict[tuple[str, str], float],   # jacobian: {(term_out, term_in): dI_out/dv_in}
    ]:
        """Evaluate terminal currents and Jacobian matrix for BJT Ebers-Moll model."""
        import math
        sign = 1.0 if self.polarity == "NPN" else -1.0
        # For NPN: vbe = vb - ve, vbc = vb - vc
        # For PNP: veb = ve - vb, vcb = vc - vb
        vbe = sign * (v_b - v_e)
        vbc = sign * (v_b - v_c)

        # BE junction
        vbe_c = min(vbe, 0.85)
        arg_be = max(min(vbe_c / self.Vt, 80.0), -80.0)
        exp_be = math.exp(arg_be)
        IF = self.Is * (exp_be - 1.0)
        gF = max((self.Is / self.Vt) * exp_be, 1e-12)
        if vbe > vbe_c:
            IF += gF * (vbe - vbe_c)

        # BC junction
        vbc_c = min(vbc, 0.85)
        arg_bc = max(min(vbc_c / self.Vt, 80.0), -80.0)
        exp_bc = math.exp(arg_bc)
        IR = self.Is * (exp_bc - 1.0)
        gR = max((self.Is / self.Vt) * exp_bc, 1e-12)
        if vbc > vbc_c:
            IR += gR * (vbc - vbc_c)

        # NPN normal currents into terminals:
        # iB = IF/beta + IR
        # iC = IF - 2*IR
        # iE = -(iB + iC)
        iB_npn = (IF / self.beta) + IR
        iC_npn = IF - 2.0 * IR
        iE_npn = -(iB_npn + iC_npn)

        # Derivatives wrt (vbe, vbc):
        # dib/dvbe = gF/beta, dib/dvbc = gR
        # dic/dvbe = gF,      dic/dvbc = -2*gR
        if self.polarity == "NPN":
            i_b, i_c, i_e = iB_npn, iC_npn, iE_npn
            dib_dvb = (gF / self.beta) + gR
            dib_dvc = -gR
            dib_dve = -(gF / self.beta)

            dic_dvb = gF - 2.0 * gR
            dic_dvc = 2.0 * gR
            dic_dve = -gF
        else:
            # PNP: currents reversed
            i_b, i_c, i_e = -iB_npn, -iC_npn, -iE_npn
            dib_dvb = (gF / self.beta) + gR
            dib_dvc = -gR
            dib_dve = -(gF / self.beta)

            dic_dvb = gF - 2.0 * gR
            dic_dvc = 2.0 * gR
            dic_dve = -gF

        die_dvb = -(dib_dvb + dic_dvb)
        die_dvc = -(dib_dvc + dic_dvc)
        die_dve = -(dib_dve + dic_dve)

        currents = {"C": i_c, "B": i_b, "E": i_e}
        jacobian = {
            ("B", "B"): dib_dvb, ("B", "C"): dib_dvc, ("B", "E"): dib_dve,
            ("C", "B"): dic_dvb, ("C", "C"): dic_dvc, ("C", "E"): dic_dve,
            ("E", "B"): die_dvb, ("E", "C"): die_dvc, ("E", "E"): die_dve,
        }
        return currents, jacobian


@dataclass
class MOSFET(_Element):
    """Metal-Oxide-Semiconductor Field-Effect Transistor (quadratic model).

    Cutoff, Triode / Linear, and Saturation regions.

    Parameters
    ----------
    name : str
        Designator (must start with 'M').
    node_d : str
        Drain node.
    node_g : str
        Gate node.
    node_s : str
        Source node.
    polarity : {"NMOS", "PMOS"}, optional
        Channel polarity (default "NMOS").
    Vth : float, optional
        Threshold voltage in Volts (default 1.0 V for NMOS, -1.0 V for PMOS).
    Kp : float, optional
        Transconductance parameter in A/V^2 (default 1e-3 A/V^2).
    lambda_param : float, optional
        Channel-length modulation in 1/V (default 0.0).
    """

    node_d: str
    node_g: str
    node_s: str
    polarity: Literal["NMOS", "PMOS"] = "NMOS"
    Vth: float = 1.0
    Kp: float = 1e-3
    lambda_param: float = 0.0

    def evaluate(
        self, v_d: float, v_g: float, v_s: float
    ) -> tuple[
        dict[str, float],              # currents: {"D": id, "G": ig, "S": is}
        dict[tuple[str, str], float],   # jacobian: {(term_out, term_in): dI_out/dv_in}
    ]:
        """Evaluate terminal currents and Jacobian matrix for MOSFET model."""
        g_min = 1e-12
        if self.polarity == "NMOS":
            vgs = v_g - v_s
            vds = v_d - v_s
            vth = self.Vth
            vov = vgs - vth
            if vov <= 0.0:
                id_val = 0.0
                gm = 0.0
                gds = g_min
            else:
                sign = 1.0 if vds >= 0.0 else -1.0
                vds_abs = abs(vds)
                mod = 1.0 + self.lambda_param * vds_abs
                if vds_abs < vov:
                    # Triode / Linear
                    id_val = sign * self.Kp * (vov * vds_abs - 0.5 * vds_abs**2) * mod
                    gm = sign * self.Kp * vds_abs * mod
                    gds = self.Kp * (vov - vds_abs) * mod + g_min
                else:
                    # Saturation
                    id_val = sign * 0.5 * self.Kp * (vov**2) * mod
                    gm = sign * self.Kp * vov * mod
                    gds = 0.5 * self.Kp * (vov**2) * self.lambda_param + g_min

            did_dvd = gds
            did_dvg = gm
            did_dvs = -(gm + gds)
            currents = {"D": id_val, "G": 0.0, "S": -id_val}
            jacobian = {
                ("D", "D"): did_dvd, ("D", "G"): did_dvg, ("D", "S"): did_dvs,
                ("G", "D"): 0.0,     ("G", "G"): g_min,   ("G", "S"): 0.0,
                ("S", "D"): -did_dvd,("S", "G"): -did_dvg,("S", "S"): -did_dvs,
            }
            return currents, jacobian
        else:
            # PMOS
            vsg = v_s - v_g
            vsd = v_s - v_d
            vth = abs(self.Vth)
            vov = vsg - vth
            if vov <= 0.0:
                is_val = 0.0
                gm = 0.0
                gds = g_min
            else:
                sign = 1.0 if vsd >= 0.0 else -1.0
                vsd_abs = abs(vsd)
                mod = 1.0 + self.lambda_param * vsd_abs
                if vsd_abs < vov:
                    # Triode / Linear
                    is_val = sign * self.Kp * (vov * vsd_abs - 0.5 * vsd_abs**2) * mod
                    gm = sign * self.Kp * vsd_abs * mod
                    gds = self.Kp * (vov - vsd_abs) * mod + g_min
                else:
                    # Saturation
                    is_val = sign * 0.5 * self.Kp * (vov**2) * mod
                    gm = sign * self.Kp * vov * mod
                    gds = 0.5 * self.Kp * (vov**2) * self.lambda_param + g_min

            # Current flows into Source, out of Drain
            dis_dvs = gm + gds
            dis_dvg = -gm
            dis_dvd = -gds
            currents = {"D": -is_val, "G": 0.0, "S": is_val}
            jacobian = {
                ("D", "D"): gds,       ("D", "G"): gm,        ("D", "S"): -(gm + gds),
                ("G", "D"): 0.0,       ("G", "G"): g_min,     ("G", "S"): 0.0,
                ("S", "D"): dis_dvd,   ("S", "G"): dis_dvg,   ("S", "S"): dis_dvs,
            }
            return currents, jacobian

