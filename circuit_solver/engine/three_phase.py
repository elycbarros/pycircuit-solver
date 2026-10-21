"""
three_phase.py — Análise e Modelagem de Sistemas Trifásicos (Three-Phase Circuits).

Cobertura completa conforme manuais canônicos (Alexander & Sadiku Cap. 12, Nilsson Cap. 11):
- Fontes balanceadas em Estrela (Y) e Triângulo (Δ) em sequência direta (ABC) e inversa (ACB).
- Cargas balanceadas e desbalanceadas Y e Δ.
- Relações fundamentais entre tensões/correntes de fase e linha.
- Medição de potência pelo Método dos Dois Wattímetros (Teorema de Blondel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from circuit_solver.core.elements import VoltageSource, Resistor, Inductor, Capacitor, _Element

__all__ = [
    "TwoWattmeterResult",
    "build_three_phase_source",
    "build_wye_load",
    "build_delta_load",
    "analyze_two_wattmeters",
]


@dataclass
class TwoWattmeterResult:
    """Results of the Two-Wattmeter power measurement method (Blondel's Theorem).

    Attributes
    ----------
    W1 : float
        Reading of Wattmeter 1 in Watts (connected to Phase A with voltage V_ac).
    W2 : float
        Reading of Wattmeter 2 in Watts (connected to Phase B with voltage V_bc).
    P_total : float
        Total 3-phase active power in Watts (W1 + W2).
    Q_total : float
        Total 3-phase reactive power in VAR (Im(Vac*conj(Ia) + Vbc*conj(Ib))).
    S_total : float
        Total 3-phase apparent power in VA (sqrt(P^2 + Q^2)).
    power_factor : float
        System power factor (P / S).
    phase_angle_deg : float
        Power factor phase angle in degrees.
    """

    W1: float
    W2: float
    P_total: float
    Q_total: float
    S_total: float
    power_factor: float
    phase_angle_deg: float


def build_three_phase_source(
    connection: Literal["Y", "DELTA"] = "Y",
    V_rms: float = 120.0,
    sequence: Literal["ABC", "ACB"] = "ABC",
    node_a: str = "a",
    node_b: str = "b",
    node_c: str = "c",
    node_neutral: str = "n",
    prefix: str = "Vs",
) -> list[VoltageSource]:
    """Generate balanced three-phase AC independent voltage source elements.

    Parameters
    ----------
    connection : {"Y", "DELTA"}, optional
        Source configuration (default "Y").
    V_rms : float, optional
        RMS voltage magnitude in Volts (Phase voltage for Y, Line voltage for DELTA).
    sequence : {"ABC", "ACB"}, optional
        Phase rotation sequence (default "ABC" - positive sequence).
    node_a, node_b, node_c : str
        Line terminal node labels.
    node_neutral : str
        Neutral point node label (used when connection is "Y").
    prefix : str
        Name designator prefix for sources.

    Returns
    -------
    list[VoltageSource]
        Three phase-shifted voltage sources with peak AC amplitudes sqrt(2)*V_rms,
        matching solve_ac. Convert solved voltage/current phasors to RMS before
        calling analyze_two_wattmeters.
    """
    if connection not in ("Y", "DELTA") or sequence not in ("ABC", "ACB"):
        raise ValueError("Expected connection Y/DELTA and sequence ABC/ACB.")
    V_peak = math.sqrt(2.0) * V_rms
    if sequence == "ABC":
        phase_b = -120.0
        phase_c = 120.0
    else:
        # Sequência inversa (ACB)
        phase_b = 120.0
        phase_c = -120.0

    if connection == "Y":
        return [
            VoltageSource(f"{prefix}_a", node_a, node_neutral, voltage=0.0, ac_mag=V_peak, ac_phase=0.0),
            VoltageSource(f"{prefix}_b", node_b, node_neutral, voltage=0.0, ac_mag=V_peak, ac_phase=phase_b),
            VoltageSource(f"{prefix}_c", node_c, node_neutral, voltage=0.0, ac_mag=V_peak, ac_phase=phase_c),
        ]
    else:
        # Delta connection (closed loop line-to-line sources)
        return [
            VoltageSource(f"{prefix}_ab", node_a, node_b, voltage=0.0, ac_mag=V_peak, ac_phase=0.0),
            VoltageSource(f"{prefix}_bc", node_b, node_c, voltage=0.0, ac_mag=V_peak, ac_phase=phase_b),
            VoltageSource(f"{prefix}_ca", node_c, node_a, voltage=0.0, ac_mag=V_peak, ac_phase=phase_c),
        ]


def build_wye_load(
    R: float = 0.0,
    L: float = 0.0,
    C: float = 0.0,
    node_a: str = "a",
    node_b: str = "b",
    node_c: str = "c",
    node_neutral: str = "n",
    prefix: str = "Zy",
) -> list[_Element]:
    """Construct a balanced three-phase Wye (Y) load branch.

    Parameters
    ----------
    R : float
        Per-phase series resistance in Ohms.
    L : float
        Per-phase series inductance in Henrys.
    C : float
        Per-phase series capacitance in Farads.
    """
    elements: list[_Element] = []
    branches = [("a", node_a), ("b", node_b), ("c", node_c)]

    for phase, node_line in branches:
        curr_node = node_line
        if R > 0 and (L > 0 or C > 0):
            mid_r = f"_{prefix}_{phase}_midR"
            elements.append(Resistor(f"{prefix}_R_{phase}", curr_node, mid_r, R))
            curr_node = mid_r
        elif R > 0:
            elements.append(Resistor(f"{prefix}_R_{phase}", curr_node, node_neutral, R))
            continue

        if L > 0 and C > 0:
            mid_l = f"_{prefix}_{phase}_midL"
            elements.append(Inductor(f"{prefix}_L_{phase}", curr_node, mid_l, L))
            curr_node = mid_l
        elif L > 0:
            elements.append(Inductor(f"{prefix}_L_{phase}", curr_node, node_neutral, L))
            continue

        if C > 0:
            elements.append(Capacitor(f"{prefix}_C_{phase}", curr_node, node_neutral, C))

    return elements


def build_delta_load(
    R: float = 0.0,
    L: float = 0.0,
    C: float = 0.0,
    node_a: str = "a",
    node_b: str = "b",
    node_c: str = "c",
    prefix: str = "Zd",
) -> list[_Element]:
    """Construct a balanced three-phase Delta (Δ) load branch."""
    elements: list[_Element] = []
    pairs = [("ab", node_a, node_b), ("bc", node_b, node_c), ("ca", node_c, node_a)]

    for pair, na, nb in pairs:
        curr_node = na
        if R > 0 and (L > 0 or C > 0):
            mid_r = f"_{prefix}_{pair}_midR"
            elements.append(Resistor(f"{prefix}_R_{pair}", curr_node, mid_r, R))
            curr_node = mid_r
        elif R > 0:
            elements.append(Resistor(f"{prefix}_R_{pair}", curr_node, nb, R))
            continue

        if L > 0 and C > 0:
            mid_l = f"_{prefix}_{pair}_midL"
            elements.append(Inductor(f"{prefix}_L_{pair}", curr_node, mid_l, L))
            curr_node = mid_l
        elif L > 0:
            elements.append(Inductor(f"{prefix}_L_{pair}", curr_node, nb, L))
            continue

        if C > 0:
            elements.append(Capacitor(f"{prefix}_C_{pair}", curr_node, nb, C))

    return elements


def analyze_two_wattmeters(
    V_line_ac: complex,
    V_line_bc: complex,
    I_line_a: complex,
    I_line_b: complex,
) -> TwoWattmeterResult:
    """Compute active, reactive and apparent power using Blondel's Two-Wattmeter method.

    Wattmeter 1: Current Ia through current coil, voltage Vac across voltage coil.
    Wattmeter 2: Current Ib through current coil, voltage Vbc across voltage coil.

    Formulas:
      W1 = Re(Vac * conj(Ia))
      W2 = Re(Vbc * conj(Ib))
      P_tot = W1 + W2
      Q_tot = Im(Vac * conj(Ia) + Vbc * conj(Ib))

    RMS phasors and a three-wire system (Ia + Ib + Ic = 0) are required.
    For four-wire loads with neutral current, use all three phase powers.
    """
    w1 = float(np.real(V_line_ac * np.conj(I_line_a)))
    w2 = float(np.real(V_line_bc * np.conj(I_line_b)))

    p_tot = w1 + w2
    # RMS phasors, three-wire system: Ic = -(Ia + Ib).
    # S = Vac*conj(Ia) + Vbc*conj(Ib), independent of balance/sequence.
    q_tot = float(np.imag(V_line_ac * np.conj(I_line_a) + V_line_bc * np.conj(I_line_b)))
    s_tot = math.hypot(p_tot, q_tot)

    pf = (p_tot / s_tot) if s_tot > 1e-12 else 1.0
    pf = max(-1.0, min(1.0, pf))
    theta_deg = math.degrees(math.acos(pf))
    if q_tot < 0:
        theta_deg = -theta_deg

    return TwoWattmeterResult(
        W1=w1,
        W2=w2,
        P_total=p_tot,
        Q_total=q_tot,
        S_total=s_tot,
        power_factor=pf,
        phase_angle_deg=theta_deg,
    )
