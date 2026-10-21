"""
ac.py — Steady-state AC phasor analysis engine (Alexander & Sadiku, Cap. 9 & 10).

Formulation
-----------
In the sinusoidal steady state, every sinusoidal voltage and current is represented
as a complex phasor:

    v(t) = V_m * cos(omega * t + phi)  <--->  V = V_m * exp(j * phi)

The MNA system becomes a set of linear equations with complex coefficients:

    A_ac * x_ac = z_ac

where impedances / admittances are:
    * Resistor:    Z_R = R,                   Y_R = 1 / R
    * Capacitor:   Z_C = 1 / (j * omega * C), Y_C = j * omega * C
    * Inductor:    Z_L = j * omega * L,       Y_L = 1 / (j * omega * L) = -j / (omega * L)

Power quantities (RMS / Peak):
    * Complex power: S = (1/2) * V * I* = P + j*Q
    * Active power (P): Re(S) in Watts [W]
    * Reactive power (Q): Im(S) in Volt-Amperes Reactive [VAR]
    * Apparent power (|S|): abs(S) in Volt-Amperes [VA]
    * Power Factor (FP): P / |S| = cos(theta_v - theta_i)
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.linalg import LinAlgError

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
    _Element,
)

__all__ = ["ACSolution", "solve_ac", "ac_sweep"]


@dataclass
class ACSolution:
    """Wrapper for AC phasor solution vector.

    Parameters
    ----------
    circuit : Circuit
    x : np.ndarray (complex128)
    frequency : float
        Frequency in Hertz (Hz).
    omega : float
        Angular frequency in rad/s.
    """

    circuit: Circuit
    x: np.ndarray
    frequency: float
    omega: float
    operating_point: Any | None = None

    # ------------------------------------------------------------------
    # Voltages
    # ------------------------------------------------------------------

    def V(self, node: str) -> complex:
        """Complex phasor voltage at node."""
        if str(node).strip() == "0":
            return 0.0 + 0.0j
        n_str = str(node).strip()
        if n_str in self.circuit.node_map:
            return complex(self.x[self.circuit.node_map[n_str]])
        for k, idx in self.circuit.node_map.items():
            if k.lower() == n_str.lower():
                return complex(self.x[idx])
        raise KeyError(f"Node '{node}' is not in circuit.")

    def V_mag(self, node: str) -> float:
        """Peak amplitude of node voltage in Volts."""
        return abs(self.V(node))

    def V_rms(self, node: str) -> float:
        """RMS amplitude of node voltage in Volts."""
        return self.V_mag(node) / np.sqrt(2.0)

    def V_phase_deg(self, node: str) -> float:
        """Phase angle of node voltage in degrees [-180, 180]."""
        return float(np.degrees(cmath.phase(self.V(node))))

    def V_diff(self, node_p: str, node_n: str) -> complex:
        """Differential complex phasor V(node_p) - V(node_n)."""
        return self.V(node_p) - self.V(node_n)

    # ------------------------------------------------------------------
    # Currents
    # ------------------------------------------------------------------

    def I_vsource(self, name: str) -> complex:
        """Current phasor through voltage source (from + to - internally)."""
        idx = self.circuit.aux_index(name)
        return complex(self.x[idx])

    def I_resistor(self, name: str) -> complex:
        """Current phasor through resistor (node_p to node_n)."""
        el = self._element_by_name(name, Resistor)
        return self.V_diff(el.node_p, el.node_n) / el.resistance

    def I_capacitor(self, name: str) -> complex:
        """Current phasor through capacitor: I = j*omega*C * V."""
        el = self._element_by_name(name, Capacitor)
        y = 1j * self.omega * el.capacitance
        return self.V_diff(el.node_p, el.node_n) * y

    def I_inductor(self, name: str) -> complex:
        """Current phasor through inductor: I = V / (j*omega*L), taking mutual coupling into account."""
        coupled_names = self.circuit.get_coupled_inductor_names()
        if name in coupled_names:
            for pair in self.circuit.get_coupled_pairs():
                if pair["ind1_name"] == name:
                    Y = pair["Gamma"] / (1j * self.omega)
                    v1 = self.V_diff(pair["p1"], pair["n1"])
                    v2 = self.V_diff(pair["p2"], pair["n2"])
                    return Y[0, 0] * v1 + Y[0, 1] * v2
                elif pair["ind2_name"] == name:
                    Y = pair["Gamma"] / (1j * self.omega)
                    v1 = self.V_diff(pair["p1"], pair["n1"])
                    v2 = self.V_diff(pair["p2"], pair["n2"])
                    return Y[1, 0] * v1 + Y[1, 1] * v2

        el = self._element_by_name(name, Inductor)
        y = 1.0 / (1j * self.omega * el.inductance)
        return self.V_diff(el.node_p, el.node_n) * y

    def I_transformer_primary(self, name: str) -> complex:
        """Current phasor entering primary dotted terminal p1."""
        for pair in self.circuit.get_coupled_pairs():
            if pair["name"] == name:
                Y = pair["Gamma"] / (1j * self.omega)
                v1 = self.V_diff(pair["p1"], pair["n1"])
                v2 = self.V_diff(pair["p2"], pair["n2"])
                return Y[0, 0] * v1 + Y[0, 1] * v2
        raise KeyError(f"Transformer '{name}' not found.")

    def I_transformer_secondary(self, name: str) -> complex:
        """Current phasor entering secondary dotted terminal p2."""
        for pair in self.circuit.get_coupled_pairs():
            if pair["name"] == name:
                Y = pair["Gamma"] / (1j * self.omega)
                v1 = self.V_diff(pair["p1"], pair["n1"])
                v2 = self.V_diff(pair["p2"], pair["n2"])
                return Y[1, 0] * v1 + Y[1, 1] * v2
        raise KeyError(f"Transformer '{name}' not found.")

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def complex_power(self, name: str) -> complex:
        """Complex power S = (1/2) * V * conj(I) absorbed by an element."""
        for el in self.circuit.elements:
            if el.name == name:
                if isinstance(el, Transformer):
                    v1 = self.V_diff(el.p1, el.n1)
                    i1 = self.I_transformer_primary(name)
                    v2 = self.V_diff(el.p2, el.n2)
                    i2 = self.I_transformer_secondary(name)
                    return 0.5 * (v1 * np.conj(i1) + v2 * np.conj(i2))
                v = self.V_diff(getattr(el, "node_p", "0"), getattr(el, "node_n", "0"))
                if isinstance(el, Resistor):
                    i = self.I_resistor(name)
                elif isinstance(el, Capacitor):
                    i = self.I_capacitor(name)
                elif isinstance(el, Inductor):
                    i = self.I_inductor(name)
                elif isinstance(el, VoltageSource):
                    # For a source, current leaving + is -I_vsource
                    i = -self.I_vsource(name)
                    return 0.5 * v * np.conj(i)
                else:
                    return 0.0j
                return 0.5 * v * np.conj(i)
        raise KeyError(f"Element '{name}' not found.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=" * 64,
            f"  pycircuit-solver — AC Phasor Solution (f = {self.frequency:.2f} Hz, ω = {self.omega:.2f} rad/s)",
            "=" * 64,
            "\n  Node Voltages (Phasors)",
            "  " + "-" * 56,
            f"  {'Node':>6}  {'Peak (V)':>10}  {'RMS (V)':>10}  {'Phase (°)':>10}  {'Cartesian (V)':>18}",
        ]
        for label, idx in sorted(self.circuit.node_map.items()):
            v = self.V(label)
            mag = abs(v)
            rms = mag / np.sqrt(2.0)
            deg = np.degrees(cmath.phase(v))
            lines.append(
                f"  {label:>6}  {mag:>10.4f}  {rms:>10.4f}  {deg:>10.2f}°  {v.real:>8.3f} + j{v.imag:>6.3f}"
            )

        lines.append("\n  Branch Currents")
        lines.append("  " + "-" * 56)
        for el in self.circuit.elements:
            if isinstance(el, (Resistor, Capacitor, Inductor)):
                if isinstance(el, Resistor):
                    i = self.I_resistor(el.name)
                elif isinstance(el, Capacitor):
                    i = self.I_capacitor(el.name)
                else:
                    i = self.I_inductor(el.name)
                mag_i = abs(i)
                deg_i = np.degrees(cmath.phase(i))
                lines.append(f"  I({el.name:>6}) = {mag_i*1e3:>10.4f} mA ∠ {deg_i:>7.2f}°")
            elif isinstance(el, Transformer):
                i1 = self.I_transformer_primary(el.name)
                i2 = self.I_transformer_secondary(el.name)
                lines.append(f"  I({el.name:>4}_p) = {abs(i1)*1e3:>10.4f} mA ∠ {np.degrees(cmath.phase(i1)):>7.2f}°")
                lines.append(f"  I({el.name:>4}_s) = {abs(i2)*1e3:>10.4f} mA ∠ {np.degrees(cmath.phase(i2)):>7.2f}°")
            elif isinstance(el, (VoltageSource, VCVS, CCVS, OpAmp)):
                j = self.I_vsource(el.name)
                lines.append(f"  I({el.name:>6}) = {abs(j)*1e3:>10.4f} mA ∠ {np.degrees(cmath.phase(j)):>7.2f}°")

        lines.append("=" * 64)
        return "\n".join(lines)

    def _element_by_name(self, name: str, kind: type) -> object:
        for el in self.circuit.elements:
            if el.name == name and isinstance(el, kind):
                return el
        raise KeyError(f"No {kind.__name__} named '{name}' in circuit.")


def solve_ac(circuit: Circuit, frequency: float, *, op: Any | None = None) -> ACSolution:
    """Solve the circuit for sinusoidal steady state at a given frequency.

    If the circuit contains non-linear semiconductor elements (Diodes, BJTs, MOSFETs),
    it is automatically linearized around its DC operating point (Q-point) using
    their small-signal hybrid-pi / conductances models.

    Parameters
    ----------
    circuit : Circuit
    frequency : float
        Frequency in Hz (frequency > 0).
    op : OperatingPoint | None, optional
        Pre-computed operating point to use for linearization.

    Returns
    -------
    ACSolution
    """
    if circuit.has_nonlinear:
        from circuit_solver.engine.small_signal import build_small_signal_circuit
        lin_circuit, op_calc = build_small_signal_circuit(circuit, op=op)
        sol = solve_ac(lin_circuit, frequency)
        sol.operating_point = op_calc
        return sol

    if frequency <= 0:
        raise ValueError(f"Frequency must be > 0 Hz, got {frequency}")

    omega = 2.0 * np.pi * frequency
    n = circuit.size
    A = np.zeros((n, n), dtype=np.complex128)
    z = np.zeros(n, dtype=np.complex128)
    coupled_names = circuit.get_coupled_inductor_names()

    def idx(node: str) -> int | None:
        return circuit.node_index(node)

    def aux(name: str) -> int:
        return circuit.aux_index(name)

    for el in circuit.elements:
        # Resistor: Y = 1/R
        if isinstance(el, Resistor):
            y = 1.0 / el.resistance
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                A[p, p] += y
            if n_node is not None:
                A[n_node, n_node] += y
            if p is not None and n_node is not None:
                A[p, n_node] -= y
                A[n_node, p] -= y

        # Capacitor: Y = j*omega*C
        elif isinstance(el, Capacitor):
            y = 1j * omega * el.capacitance
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                A[p, p] += y
            if n_node is not None:
                A[n_node, n_node] += y
            if p is not None and n_node is not None:
                A[p, n_node] -= y
                A[n_node, p] -= y

        # Inductor: Y = 1 / (j*omega*L) = -j / (omega*L)
        elif isinstance(el, Inductor):
            if el.name in coupled_names:
                continue
            y = 1.0 / (1j * omega * el.inductance)
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                A[p, p] += y
            if n_node is not None:
                A[n_node, n_node] += y
            if p is not None and n_node is not None:
                A[p, n_node] -= y
                A[n_node, p] -= y

        # VoltageSource (AC phasor)
        elif isinstance(el, VoltageSource):
            k = aux(el.name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                A[p, k] += 1.0
            if n_node is not None:
                A[n_node, k] -= 1.0
            if p is not None:
                A[k, p] += 1.0
            if n_node is not None:
                A[k, n_node] -= 1.0

            # Phasor value: if ac_mag is defined, use it; else fallback to voltage
            mag = el.ac_mag if el.ac_mag != 0.0 else el.voltage
            phase_rad = np.radians(el.ac_phase)
            z[k] = mag * np.exp(1j * phase_rad)

        # CurrentSource (AC phasor)
        elif isinstance(el, CurrentSource):
            p, n_node = idx(el.node_p), idx(el.node_n)
            mag = el.ac_mag if el.ac_mag != 0.0 else el.current
            phase_rad = np.radians(el.ac_phase)
            val = mag * np.exp(1j * phase_rad)
            if p is not None:
                z[p] += val
            if n_node is not None:
                z[n_node] -= val

        # VCVS
        elif isinstance(el, VCVS):
            k = aux(el.name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            cp, cn = idx(el.ctrl_p), idx(el.ctrl_n)
            if p is not None:
                A[p, k] += 1.0
            if n_node is not None:
                A[n_node, k] -= 1.0
            if p is not None:
                A[k, p] += 1.0
            if n_node is not None:
                A[k, n_node] -= 1.0
            if cp is not None:
                A[k, cp] -= el.gain
            if cn is not None:
                A[k, cn] += el.gain
            z[k] = 0.0

        # VCCS
        elif isinstance(el, VCCS):
            gm = el.transconductance
            p, n_node = idx(el.node_p), idx(el.node_n)
            cp, cn = idx(el.ctrl_p), idx(el.ctrl_n)
            if p is not None and cp is not None:
                A[p, cp] -= gm
            if p is not None and cn is not None:
                A[p, cn] += gm
            if n_node is not None and cp is not None:
                A[n_node, cp] += gm
            if n_node is not None and cn is not None:
                A[n_node, cn] -= gm

        # CCCS
        elif isinstance(el, CCCS):
            k_vctrl = aux(el.vctrl_name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                A[p, k_vctrl] += el.gain
            if n_node is not None:
                A[n_node, k_vctrl] -= el.gain

        # CCVS
        elif isinstance(el, CCVS):
            k = aux(el.name)
            k_vctrl = aux(el.vctrl_name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                A[p, k] += 1.0
            if n_node is not None:
                A[n_node, k] -= 1.0
            if p is not None:
                A[k, p] += 1.0
            if n_node is not None:
                A[k, n_node] -= 1.0
            A[k, k_vctrl] += el.transresistance
            z[k] = 0.0

        # OpAmp
        elif isinstance(el, OpAmp):
            k = aux(el.name)
            out = idx(el.out)
            gnd = idx(el.gnd)
            in_p = idx(el.in_p)
            in_n = idx(el.in_n)
            if out is not None:
                A[out, k] += 1.0
            if gnd is not None:
                A[gnd, k] -= 1.0
            if in_p is not None:
                A[k, in_p] += 1.0
            if in_n is not None:
                A[k, in_n] -= 1.0
            z[k] = 0.0

    # Stamp coupled inductor pairs and transformers
    for pair in circuit.get_coupled_pairs():
        Y = pair["Gamma"] / (1j * omega)
        p1, n1 = idx(pair["p1"]), idx(pair["n1"])
        p2, n2 = idx(pair["p2"]), idx(pair["n2"])

        # Port 1 self
        y11 = Y[0, 0]
        if p1 is not None: A[p1, p1] += y11
        if n1 is not None: A[n1, n1] += y11
        if p1 is not None and n1 is not None:
            A[p1, n1] -= y11
            A[n1, p1] -= y11

        # Port 2 self
        y22 = Y[1, 1]
        if p2 is not None: A[p2, p2] += y22
        if n2 is not None: A[n2, n2] += y22
        if p2 is not None and n2 is not None:
            A[p2, n2] -= y22
            A[n2, p2] -= y22

        # Mutual cross
        y12 = Y[0, 1]
        y21 = Y[1, 0]
        if p1 is not None and p2 is not None: A[p1, p2] += y12
        if p1 is not None and n2 is not None: A[p1, n2] -= y12
        if n1 is not None and p2 is not None: A[n1, p2] -= y12
        if n1 is not None and n2 is not None: A[n1, n2] += y12

        if p2 is not None and p1 is not None: A[p2, p1] += y21
        if p2 is not None and n1 is not None: A[p2, n1] -= y21
        if n2 is not None and p1 is not None: A[n2, p1] -= y21
        if n2 is not None and n1 is not None: A[n2, n1] += y21

    try:
        x = np.linalg.solve(A, z)
    except LinAlgError as exc:
        raise LinAlgError(f"AC MNA matrix is singular at f={frequency} Hz: {exc}") from exc

    return ACSolution(circuit, x, frequency, omega)


def ac_sweep(
    circuit: Circuit,
    frequencies: np.ndarray,
    target_node: str,
    *,
    op: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a frequency sweep and return (freqs, magnitude_dB, phase_deg).

    Automatically linearizes non-linear semiconductor models once around the
    DC Q-point prior to the sweep for high execution performance.

    Parameters
    ----------
    circuit : Circuit
    frequencies : np.ndarray
        Array of frequencies in Hz.
    target_node : str
        Node to measure.
    op : OperatingPoint | None, optional
        Pre-computed operating point to use for linearization.

    Returns
    -------
    tuple of (frequencies, mag_dB, phase_deg)
    """
    if circuit.has_nonlinear:
        from circuit_solver.engine.small_signal import build_small_signal_circuit
        circuit, _ = build_small_signal_circuit(circuit, op=op)

    mags_db = []
    phases_deg = []

    for f in frequencies:
        sol = solve_ac(circuit, f)
        v = sol.V(target_node)
        mag = abs(v)
        # Convert to dB (avoid log(0))
        db = 20.0 * np.log10(max(mag, 1e-12))
        deg = np.degrees(cmath.phase(v))
        mags_db.append(db)
        phases_deg.append(deg)

    return frequencies, np.array(mags_db), np.array(phases_deg)
