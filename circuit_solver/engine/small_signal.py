"""
small_signal.py — Non-linear semiconductor small-signal linearization and Q-point analysis.

This module bridges DC non-linear operating point calculation (Newton-Raphson)
with linear AC frequency-domain phasor analysis (Bode diagrams, small-signal gain).

Theory & Small-Signal Models (Sedra & Smith, Alexander & Sadiku, Neamen):
------------------------------------------------------------------------
1. Diode:
   * Q-point: V_D, I_D = Is * (exp(V_D / (n*Vt)) - 1)
   * Dynamic conductance: g_d = dI_D/dV_D = (I_D + Is) / (n * Vt)
   * Dynamic resistance:  r_d = 1 / g_d

2. BJT (NPN & PNP — Hybrid-pi Giacoleto Model):
   * Q-point: V_BE, V_CE, I_B, I_C
   * Operating region: Cutoff (V_BE < 0.5V), Saturation (V_CE < 0.2V), Forward Active
   * Transconductance:     g_m = |I_C| / Vt
   * Input resistance:     r_pi = beta / g_m = Vt / |I_B|
   * Output resistance:    r_o = (VA + |V_CE|) / |I_C|  (default ~100k or from Early effect)
   * AC stamps:
     - Resistor r_pi between Base and Emitter.
     - VCCS g_m * v_be from Collector to Emitter.
     - Resistor r_o between Collector and Emitter.

3. MOSFET (NMOS & PMOS):
   * Q-point: V_GS, V_DS, I_D, V_ov = |V_GS| - |Vth|
   * Operating region: Cutoff (V_ov <= 0), Triode (V_DS < V_ov), Saturation (V_DS >= V_ov)
   * Transconductance:
     - Saturated: g_m = Kp * V_ov * (1 + lambda * V_DS) = sqrt(2 * Kp * I_D * (1 + lambda*V_DS))
     - Triode:    g_m = Kp * V_DS * (1 + lambda * V_DS)
   * Output resistance:    r_ds = 1 / g_ds where g_ds = lambda * I_D
   * AC stamps:
     - VCCS g_m * v_gs from Drain to Source.
     - Resistor r_ds between Drain and Source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    _Element,
)
from circuit_solver.core.solution import Solution


@dataclass
class DiodeQPoint:
    """Operating point and small-signal parameters for a Diode."""
    name: str
    node_p: str
    node_n: str
    v_d: float
    i_d: float
    g_d: float
    r_d: float
    state: str  # "FORWARD", "REVERSE", "BREAKDOWN"


@dataclass
class BJTQPoint:
    """Operating point and hybrid-pi small-signal parameters for a BJT."""
    name: str
    polarity: str
    node_c: str
    node_b: str
    node_e: str
    v_be: float
    v_ce: float
    v_bc: float
    i_b: float
    i_c: float
    i_e: float
    beta: float
    region: str  # "FORWARD ACTIVE", "SATURATION", "CUTOFF"
    g_m: float
    r_pi: float
    r_o: float


@dataclass
class MOSFETQPoint:
    """Operating point and small-signal parameters for a MOSFET."""
    name: str
    polarity: str
    node_d: str
    node_g: str
    node_s: str
    v_gs: float
    v_ds: float
    v_ov: float
    i_d: float
    v_th: float
    region: str  # "SATURATION", "TRIODE", "CUTOFF"
    g_m: float
    r_ds: float


class OperatingPoint:
    """Holds the DC bias state and small-signal parameters for all semiconductors."""

    def __init__(self, dc_solution: Solution, circuit: Circuit) -> None:
        self.dc_solution = dc_solution
        self.circuit = circuit
        self.diodes: dict[str, DiodeQPoint] = {}
        self.bjts: dict[str, BJTQPoint] = {}
        self.mosfets: dict[str, MOSFETQPoint] = {}
        self._extract_parameters()

    def _extract_parameters(self) -> None:
        sol = self.dc_solution

        for el in self.circuit.elements:
            # ---------------------------------------------------------------
            # 1. Diode / Zener
            # ---------------------------------------------------------------
            if isinstance(el, (Diode, ZenerDiode)):
                vd = sol.V_diff(el.node_p, el.node_n)
                id_val, gd = el.evaluate(vd)
                rd = 1.0 / max(gd, 1e-12)

                if isinstance(el, ZenerDiode) and vd <= -el.Vz:
                    state = "BREAKDOWN (ZENER)"
                elif vd >= 0.5:
                    state = "FORWARD BIASED"
                else:
                    state = "REVERSE BIASED"

                self.diodes[el.name] = DiodeQPoint(
                    name=el.name,
                    node_p=el.node_p,
                    node_n=el.node_n,
                    v_d=vd,
                    i_d=id_val,
                    g_d=gd,
                    r_d=rd,
                    state=state,
                )

            # ---------------------------------------------------------------
            # 2. BJT
            # ---------------------------------------------------------------
            elif isinstance(el, BJT):
                vc = sol.V(el.node_c)
                vb = sol.V(el.node_b)
                ve = sol.V(el.node_e)
                sign = 1.0 if el.polarity == "NPN" else -1.0

                vbe = sign * (vb - ve)
                vce = sign * (vc - ve)
                vbc = sign * (vb - vc)

                currents, jacobian = el.evaluate(vc, vb, ve)
                ib = abs(currents["B"])
                ic = abs(currents["C"])
                ie = abs(currents["E"])

                # Determine region
                if vbe < 0.5:
                    region = "CUTOFF"
                elif vce < 0.25 and ib > 1e-9:
                    region = "SATURATION"
                else:
                    region = "FORWARD ACTIVE"

                # Small-signal hybrid-pi parameters
                vt = el.Vt
                # gm = |Ic| / Vt
                gm = max(ic / vt, 1e-12)
                # r_pi = beta / gm = Vt / |Ib|
                r_pi = max(el.beta / gm, 1.0)
                # ro from output conductance or default Early effect
                g_ce = max(abs(jacobian.get(("C", "C"), 0.0)), 1e-6)
                r_o = 1.0 / g_ce

                self.bjts[el.name] = BJTQPoint(
                    name=el.name,
                    polarity=el.polarity,
                    node_c=el.node_c,
                    node_b=el.node_b,
                    node_e=el.node_e,
                    v_be=vbe,
                    v_ce=vce,
                    v_bc=vbc,
                    i_b=ib,
                    i_c=ic,
                    i_e=ie,
                    beta=el.beta,
                    region=region,
                    g_m=gm,
                    r_pi=r_pi,
                    r_o=r_o,
                )

            # ---------------------------------------------------------------
            # 3. MOSFET
            # ---------------------------------------------------------------
            elif isinstance(el, MOSFET):
                vd = sol.V(el.node_d)
                vg = sol.V(el.node_g)
                vs = sol.V(el.node_s)

                if el.polarity == "NMOS":
                    vgs = vg - vs
                    vds = vd - vs
                    vov = vgs - el.Vth
                else:
                    vgs = vs - vg
                    vds = vs - vd
                    vov = vgs - abs(el.Vth)

                currents, jacobian = el.evaluate(vd, vg, vs)
                id_val = abs(currents["D"])

                if vov <= 0.0:
                    region = "CUTOFF"
                    gm = 0.0
                    rds = 1e9
                elif vds < vov:
                    region = "TRIODE / LINEAR"
                    gm = max(el.Kp * vds * (1.0 + el.lambda_param * vds), 1e-12)
                    gds = max(el.Kp * (vov - vds) * (1.0 + el.lambda_param * vds) + el.lambda_param * id_val, 1e-9)
                    rds = 1.0 / gds
                else:
                    region = "SATURATION"
                    gm = max(el.Kp * vov * (1.0 + el.lambda_param * vds), 1e-12)
                    gds = max(el.lambda_param * id_val, 1e-8)
                    rds = 1.0 / gds

                self.mosfets[el.name] = MOSFETQPoint(
                    name=el.name,
                    polarity=el.polarity,
                    node_d=el.node_d,
                    node_g=el.node_g,
                    node_s=el.node_s,
                    v_gs=vgs,
                    v_ds=vds,
                    v_ov=vov,
                    i_d=id_val,
                    v_th=el.Vth,
                    region=region,
                    g_m=gm,
                    r_ds=rds,
                )

    def summary(self) -> str:
        """Formatted pedagogical operating point and small-signal parameters report."""
        lines = [
            "=" * 78,
            " 📊 RELATÓRIO DO PONTO DE OPERAÇÃO DC (Q-POINT) E PARÂMETROS DE PEQUENOS SINAIS",
            "=" * 78,
        ]

        if not self.diodes and not self.bjts and not self.mosfets:
            lines.append("  Nenhum componente semicondutor presente no circuito.")
            lines.append("=" * 78)
            return "\n".join(lines)

        # Diodes
        if self.diodes:
            lines.append("\n1. DIODOS E ZENERS:")
            lines.append("  " + "-" * 74)
            lines.append(f"  {'Nome':<8} {'Estado':<18} {'V_D (V)':>10} {'I_D (mA)':>12} {'r_d (Ω)':>12} {'g_d (mA/V)':>12}")
            lines.append("  " + "-" * 74)
            for d in self.diodes.values():
                lines.append(
                    f"  {d.name:<8} {d.state:<18} {d.v_d:>10.4f} {d.i_d*1e3:>12.4f} {d.r_d:>12.2f} {d.g_d*1e3:>12.4f}"
                )

        # BJTs
        if self.bjts:
            lines.append("\n2. TRANSISTORES BIPOLARES (BJT — Modelo Híbrido-π):")
            lines.append("  " + "-" * 74)
            for b in self.bjts.values():
                lines.append(f"  • {b.name} [{b.polarity}] — Região: {b.region}")
                lines.append(f"    Polarização DC:  V_BE = {b.v_be:.4f} V | V_CE = {b.v_ce:.4f} V | V_BC = {b.v_bc:.4f} V")
                lines.append(f"    Correntes DC:    I_B = {b.i_b*1e6:.2f} μA | I_C = {b.i_c*1e3:.4f} mA | I_E = {b.i_e*1e3:.4f} mA (β = {b.beta:.1f})")
                lines.append(f"    Pequenos Sinais: g_m = {b.g_m*1e3:.3f} mA/V | r_π = {b.r_pi:.1f} Ω | r_o = {b.r_o/1e3:.1f} kΩ")
                lines.append("")

        # MOSFETs
        if self.mosfets:
            lines.append("\n3. TRANSISTORES DE EFEITO DE CAMPO (MOSFET):")
            lines.append("  " + "-" * 74)
            for m in self.mosfets.values():
                lines.append(f"  • {m.name} [{m.polarity}] — Região: {m.region}")
                lines.append(f"    Polarização DC:  V_GS = {m.v_gs:.4f} V | V_DS = {m.v_ds:.4f} V | V_ov = {m.v_ov:.4f} V (V_th = {m.v_th:.2f} V)")
                lines.append(f"    Corrente DC:     I_D = {m.i_d*1e3:.4f} mA")
                lines.append(f"    Pequenos Sinais: g_m = {m.g_m*1e3:.3f} mA/V | r_ds = {m.r_ds/1e3:.2f} kΩ")
                lines.append("")

        lines.extend([
            "💡 NOTA TEÓRICA:",
            "  Os parâmetros de pequenos sinais (g_m, r_π, r_o, r_ds) são as derivadas",
            "  do Jacobiano linearizado no ponto Q estático. O solver fasorial CA usará",
            "  esses valores para montar a matriz de admitância de pequenos sinais.",
            "=" * 78,
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small-Signal Operating Point Solver & Circuit Linearizer
# ---------------------------------------------------------------------------

def compute_operating_point(circuit: Circuit) -> OperatingPoint:
    """Solve the DC operating point and extract small-signal parameters for semiconductors.

    Parameters
    ----------
    circuit : Circuit
        The circuit with semiconductor models and bias sources.

    Returns
    -------
    OperatingPoint
        Object containing all Q-point voltages, currents, and dynamic parameters.
    """
    from circuit_solver.engine.nonlinear import solve_nonlinear
    from circuit_solver.engine.mna import solve

    if circuit.has_nonlinear:
        dc_sol = solve_nonlinear(circuit)
    else:
        dc_sol = solve(circuit)

    return OperatingPoint(dc_sol, circuit)


def build_small_signal_circuit(circuit: Circuit, op: OperatingPoint | None = None) -> tuple[Circuit, OperatingPoint]:
    """Construct an equivalent linear Circuit where semiconductors are replaced
    by their small-signal linear models (resistors, VCCS sources), and pure DC
    bias sources are deactivated.

    Parameters
    ----------
    circuit : Circuit
        Original circuit containing semiconductors, DC bias, and AC signal sources.
    op : OperatingPoint | None, optional
        Pre-computed operating point. If None, it will be automatically computed.

    Returns
    -------
    tuple[Circuit, OperatingPoint]
        The linearized equivalent small-signal Circuit, and the OperatingPoint data.
    """
    if op is None:
        op = compute_operating_point(circuit)

    lin_elements: list[_Element] = []

    for el in circuit.elements:
        # Passives: R, C, L, Transformers, VCVS, VCCS, CCCS, CCVS, OpAmp
        if isinstance(el, (Resistor, Capacitor, Inductor, Transformer, MutualCoupling, VCVS, VCCS, CCCS, CCVS, OpAmp)):
            lin_elements.append(el)

        # Independent Voltage Sources
        elif isinstance(el, VoltageSource):
            # If it has an AC specification, keep it as AC source (DC value zeroed in small-signal)
            if el.ac_mag != 0.0:
                lin_elements.append(
                    VoltageSource(
                        el.name, el.node_p, el.node_n, 0.0,
                        ac_mag=el.ac_mag, ac_phase=el.ac_phase
                    )
                )
            else:
                # Pure DC bias source: in small-signal AC, it becomes an AC short-circuit (0 V)
                lin_elements.append(
                    VoltageSource(
                        el.name, el.node_p, el.node_n, 0.0,
                        ac_mag=0.0, ac_phase=0.0
                    )
                )

        # Independent Current Sources
        elif isinstance(el, CurrentSource):
            if el.ac_mag != 0.0:
                lin_elements.append(
                    CurrentSource(
                        el.name, el.node_p, el.node_n, 0.0,
                        ac_mag=el.ac_mag, ac_phase=el.ac_phase
                    )
                )
            # Pure DC current sources become open-circuits in AC: omitted

        # Diodes: replaced by dynamic resistance r_d
        elif isinstance(el, (Diode, ZenerDiode)):
            d_qp = op.diodes[el.name]
            r_d_val = max(d_qp.r_d, 1e-4)
            lin_elements.append(
                Resistor(f"{el.name}_rd", el.node_p, el.node_n, r_d_val)
            )

        # BJT: replaced by hybrid-pi model (r_pi, VCCS g_m, r_o)
        elif isinstance(el, BJT):
            b_qp = op.bjts[el.name]
            # 1. Base-Emitter input resistor r_pi
            lin_elements.append(
                Resistor(f"{el.name}_rpi", el.node_b, el.node_e, max(b_qp.r_pi, 1e-2))
            )
            # 2. Collector-Emitter controlled current source gm * vbe
            # Current flows into Collector and out of Emitter
            lin_elements.append(
                VCCS(f"{el.name}_gm", el.node_c, el.node_e, el.node_b, el.node_e, b_qp.g_m)
            )
            # 3. Collector-Emitter output resistor r_o
            lin_elements.append(
                Resistor(f"{el.name}_ro", el.node_c, el.node_e, max(b_qp.r_o, 1.0))
            )

        # MOSFET: replaced by g_m * vgs and r_ds
        elif isinstance(el, MOSFET):
            m_qp = op.mosfets[el.name]
            # 1. Drain-Source controlled current source gm * vgs
            lin_elements.append(
                VCCS(f"{el.name}_gm", el.node_d, el.node_s, el.node_g, el.node_s, m_qp.g_m)
            )
            # 2. Drain-Source output resistor r_ds
            lin_elements.append(
                Resistor(f"{el.name}_rds", el.node_d, el.node_s, max(m_qp.r_ds, 1.0))
            )

    lin_circuit = Circuit(lin_elements, hierarchy=getattr(circuit, "hierarchy", None))
    return lin_circuit, op
