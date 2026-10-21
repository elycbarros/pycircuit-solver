"""
sensitivity.py — Analytical Sensitivity, Worst-Case (WCA) and Extreme Value Analysis (EVA).

Implements:
1. Adjoint Network Method (Tellegen's theorem / Director & Rohrer):
   Computes exact analytical sensitivities d(V_target)/d(p_i) for ALL circuit
   components in O(1) back-solves after a single LU factorization of A^T.
2. Normalized relative sensitivity S = (p / V) * (dV / dp) [% variation per % tolerance].
3. Extreme Value Analysis (EVA / Worst-Case Analysis):
   Estimates parameter corner directions from nominal derivatives (+ or -) that maximize and
   minimize the target node voltage. Evaluates both:
     - Exact non-linear corner simulation via MNA re-solve.
     - First-order linear Taylor worst-case estimate.
4. Root-Sum-Square (RSS) first-order tolerance combination (no probability coverage implied):
     Delta_V_RSS = sqrt(sum( (dV/dp_i * delta_p_i)^2 ))
5. Pedagogical Pareto ranking and White-Box formatted reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    CCCS,
    CCVS,
    OpAmp,
)
from circuit_solver.engine.mna import build_mna, solve

__all__ = [
    "ComponentSensitivity",
    "SensitivityResult",
    "analyze_sensitivity",
    "worst_case_analysis",
]


@dataclass
class ComponentSensitivity:
    """Sensitivity metrics for a single circuit component."""
    name: str
    element_type: str
    nominal_value: float
    unit: str
    tolerance: float
    abs_sensitivity: float   # d(V_target) / d(param) in [V / unit]
    rel_sensitivity: float   # (param / V_target) * (dV / d_param) [% / %]
    worst_case_dir: int      # +1 if increasing param increases V, -1 if decreases V
    param_wc_max: float      # Component value that maximizes target
    param_wc_min: float      # Component value that minimizes target
    linear_delta: float      # |dV/dp| * (tolerance * nominal)
    percent_contribution: float = 0.0  # Fraction of total absolute linear impact (Pareto)


@dataclass
class SensitivityResult:
    """Comprehensive container for sensitivity and worst-case analysis results."""
    circuit: Circuit
    target_node: str
    nominal_value: float
    components: dict[str, ComponentSensitivity]
    worst_case_max: float        # MNA solve at sensitivity-selected upper corner
    worst_case_min: float        # MNA solve at sensitivity-selected lower corner
    linear_worst_case_max: float # Nominal + sum(linear_delta)
    linear_worst_case_min: float # Nominal - sum(linear_delta)
    rss_max: float               # Nominal + sqrt(sum(linear_delta^2))
    rss_min: float               # Nominal - sqrt(sum(linear_delta^2))

    def pareto_ranking(self) -> list[ComponentSensitivity]:
        """Return components sorted by their absolute impact on the target node."""
        return sorted(
            self.components.values(),
            key=lambda c: abs(c.linear_delta),
            reverse=True,
        )

    def summary(self) -> str:
        """Format an academic and professional White-Box report."""
        lines = [
            "=" * 82,
            f"  Worst-Case (EVA) & Sensitivity Analysis Report: Node '{self.target_node}'",
            "=" * 82,
            f"  Nominal Voltage:         {self.nominal_value:>12.6f} V",
            f"  Selected Corner Range:   [{self.worst_case_min:>11.6f} V, {self.worst_case_max:>11.6f} V]  (Corner re-solve; no global bound)",
            f"  Linear Taylor Bounds:    [{self.linear_worst_case_min:>11.6f} V, {self.linear_worst_case_max:>11.6f} V]  (ΔV_wc = ±{self.linear_worst_case_max - self.nominal_value:.6f} V)",
            f"  Linear RSS Range:  [{self.rss_min:>11.6f} V, {self.rss_max:>11.6f} V]  (ΔV_rss = ±{self.rss_max - self.nominal_value:.6f} V)",
            "-" * 82,
            f"  {'Component':<10} {'Nominal':>12} {'Tol (%)':>8} {'∂V/∂p (V/u)':>14} {'S (%/%)':>10} {'Dir':>5} {'Impact (mV)':>12} {'Pareto':>8}",
            "-" * 82,
        ]

        ranking = self.pareto_ranking()
        for c in ranking:
            nom_str = f"{c.nominal_value:.4g} {c.unit}"
            tol_str = f"±{c.tolerance * 100:.1f}%"
            abs_str = f"{c.abs_sensitivity:+.4e}"
            rel_str = f"{c.rel_sensitivity:+.4f}"
            dir_str = "▲ (+)" if c.worst_case_dir > 0 else ("▼ (-)" if c.worst_case_dir < 0 else "0")
            impact_mv = f"{c.linear_delta * 1000:>10.3f}"
            pareto_str = f"{c.percent_contribution:>6.1f}%"

            lines.append(
                f"  {c.name:<10} {nom_str:>12} {tol_str:>8} {abs_str:>14} {rel_str:>10} {dir_str:>5} {impact_mv:>12} {pareto_str:>8}"
            )

        lines.extend([
            "=" * 82,
            "  * RSS assumes independent variations; it is not automatically a 3-sigma interval.",
            "    Uniform ±tolerances give sigma(V) ≈ RSS/sqrt(3) in the linear model.",
            "  * Legend: S (%/%) = (p/V)*(∂V/∂p). Dir indicates whether increasing the parameter",
            "    increases (▲) or decreases (▼) the target voltage V(node).",
            "=" * 82,
        ])
        return "\n".join(lines)


def _clone_circuit_with_overrides(circuit: Circuit, overrides: dict[str, float]) -> Circuit:
    """Clone circuit with specific component parameter values."""
    new_elements = []
    for el in circuit.elements:
        if el.name in overrides:
            val = overrides[el.name]
            if isinstance(el, Resistor):
                new_elements.append(Resistor(el.name, el.node_p, el.node_n, val, el.tolerance))
            elif isinstance(el, VoltageSource):
                new_elements.append(VoltageSource(el.name, el.node_p, el.node_n, val, el.ac_mag, el.ac_phase, el.waveform))
            elif isinstance(el, CurrentSource):
                new_elements.append(CurrentSource(el.name, el.node_p, el.node_n, val, el.ac_mag, el.ac_phase, el.waveform))
            elif isinstance(el, VCVS):
                new_elements.append(VCVS(el.name, el.node_p, el.node_n, el.ctrl_p, el.ctrl_n, val))
            elif isinstance(el, VCCS):
                new_elements.append(VCCS(el.name, el.node_p, el.node_n, el.ctrl_p, el.ctrl_n, val))
            elif isinstance(el, CCCS):
                new_elements.append(CCCS(el.name, el.node_p, el.node_n, el.vctrl_name, val))
            elif isinstance(el, CCVS):
                new_elements.append(CCVS(el.name, el.node_p, el.node_n, el.vctrl_name, val))
            else:
                new_elements.append(el)
        else:
            new_elements.append(el)
    return Circuit(new_elements)


def analyze_sensitivity(
    circuit: Circuit,
    target_node: str,
    *,
    custom_tolerances: dict[str, float] | None = None,
    default_resistor_tolerance: float = 0.05,
    default_source_tolerance: float = 0.01,
) -> SensitivityResult:
    """Compute exact analytical sensitivities and Worst-Case/EVA bounds.

    Parameters
    ----------
    circuit : Circuit
        Pre-validated circuit topology.
    target_node : str
        Node name of the voltage of interest (e.g., '2', 'out').
    custom_tolerances : dict[str, float], optional
        Mapping of element names to fractional tolerances (e.g. {'R1': 0.01, 'V1': 0.02}).
    default_resistor_tolerance : float, optional
        Default tolerance used for resistors without an explicit override (default 0.05 = 5%).
    default_source_tolerance : float, optional
        Default tolerance used for sources (default 0.01 = 1%).

    Returns
    -------
    SensitivityResult
        Complete analytical sensitivities, Pareto rankings, and EVA bounds.
    """
    supported = (Resistor, VoltageSource, CurrentSource, VCVS, VCCS, CCCS, CCVS, OpAmp)
    if any(not isinstance(el, supported) for el in circuit.elements):
        raise ValueError("Sensitivity requires a linear DC circuit of supported elements.")
    target = str(target_node).strip()
    if target == "0":
        raise ValueError("Target node cannot be ground ('0').")

    target_idx = circuit.node_index(target)
    if target_idx is None:
        raise KeyError(f"Node '{target}' not found in circuit. Available: {list(circuit.nodes)}")

    # 1. Solve nominal circuit: A * x = z
    sol_nom = solve(circuit)
    v_target_nom = sol_nom.V(target)
    x_nom = sol_nom._x
    A, _ = build_mna(circuit)

    # 2. Solve Adjoint system: A^T * y = e_target
    e_target = np.zeros(circuit.size, dtype=float)
    e_target[target_idx] = 1.0

    try:
        y_adj = np.linalg.solve(A.T, e_target)
    except np.linalg.LinAlgError as err:
        raise RuntimeError(f"Failed to solve adjoint network system: {err}") from err

    def y_node(node: str) -> float:
        idx = circuit.node_index(node)
        return float(y_adj[idx]) if idx is not None else 0.0

    def x_node(node: str) -> float:
        idx = circuit.node_index(node)
        return float(x_nom[idx]) if idx is not None else 0.0

    components: dict[str, ComponentSensitivity] = {}
    tolerances = custom_tolerances or {}

    for el in circuit.elements:
        tol = tolerances.get(el.name)

        # ------------------------------------------------------------------
        # Resistor: dV/dR = (y_p - y_n) * (x_p - x_n) / R^2
        # ------------------------------------------------------------------
        if isinstance(el, Resistor):
            if tol is None:
                tol = el.tolerance if el.tolerance > 0 else default_resistor_tolerance

            r_val = float(el.resistance)
            yp = y_node(el.node_p)
            yn = y_node(el.node_n)
            xp = x_node(el.node_p)
            xn = x_node(el.node_n)

            # Exact adjoint formula for resistor sensitivity
            # Note: G = 1/R -> dG/dR = -1/R^2. Since [A]_R = G, dA/dR = -1/R^2
            # dV/dR = y^T * (-dA/dR) * x = (yp - yn) * (xp - xn) / R^2
            d_v_d_p = (yp - yn) * (xp - xn) / (r_val ** 2)
            unit = "Ω"
            nom_val = r_val

        # ------------------------------------------------------------------
        # Voltage Source: dV/dV_s = y_aux
        # ------------------------------------------------------------------
        elif isinstance(el, VoltageSource):
            if tol is None:
                tol = default_source_tolerance

            aux_k = circuit.aux_index(el.name)
            d_v_d_p = float(y_adj[aux_k])
            unit = "V"
            nom_val = float(el.voltage)

        # ------------------------------------------------------------------
        # Current Source: dV/dI_s = y_p - y_n
        # ------------------------------------------------------------------
        elif isinstance(el, CurrentSource):
            if tol is None:
                tol = default_source_tolerance

            yp = y_node(el.node_p)
            yn = y_node(el.node_n)
            d_v_d_p = float(yp - yn)
            unit = "A"
            nom_val = float(el.current)

        # ------------------------------------------------------------------
        # VCVS: dV/d_gain = y_aux * (x_cp - x_cn)
        # ------------------------------------------------------------------
        elif isinstance(el, VCVS):
            if tol is None:
                tol = 0.0  # Typically exact unless specified

            aux_k = circuit.aux_index(el.name)
            xcp = x_node(el.ctrl_p)
            xcn = x_node(el.ctrl_n)
            d_v_d_p = float(y_adj[aux_k] * (xcp - xcn))
            unit = "V/V"
            nom_val = float(el.gain)

        # ------------------------------------------------------------------
        # VCCS: dV/d_gm = (y_p - y_n) * (x_cp - x_cn)
        # ------------------------------------------------------------------
        elif isinstance(el, VCCS):
            if tol is None:
                tol = 0.0

            yp = y_node(el.node_p)
            yn = y_node(el.node_n)
            xcp = x_node(el.ctrl_p)
            xcn = x_node(el.ctrl_n)
            d_v_d_p = float((yp - yn) * (xcp - xcn))
            unit = "S"
            nom_val = float(el.transconductance)

        else:
            # Multi-terminal elements without variable parameters skipped
            continue

        if not np.isfinite(tol) or tol < 0 or (isinstance(el, Resistor) and tol >= 1):
            raise ValueError(f"Invalid tolerance for {el.name}: {tol}")

        # Relative / Normalized sensitivity S = (p / V) * (dV/dp)
        rel_sens = (nom_val / v_target_nom * d_v_d_p) if abs(v_target_nom) > 1e-15 else 0.0

        # Direction of worst-case
        if abs(d_v_d_p) < 1e-14:
            wc_dir = 0
            p_max = nom_val
            p_min = nom_val
        elif d_v_d_p > 0:
            wc_dir = 1
            p_max = nom_val + abs(nom_val) * tol
            p_min = nom_val - abs(nom_val) * tol
        else:
            wc_dir = -1
            p_max = nom_val - abs(nom_val) * tol
            p_min = nom_val + abs(nom_val) * tol

        # First-order linear Taylor contribution
        linear_delta = abs(d_v_d_p) * (tol * abs(nom_val))

        components[el.name] = ComponentSensitivity(
            name=el.name,
            element_type=type(el).__name__,
            nominal_value=nom_val,
            unit=unit,
            tolerance=tol,
            abs_sensitivity=d_v_d_p,
            rel_sensitivity=rel_sens,
            worst_case_dir=wc_dir,
            param_wc_max=p_max,
            param_wc_min=p_min,
            linear_delta=linear_delta,
        )

    # Compute Pareto percentage contribution
    sum_linear = sum(c.linear_delta for c in components.values())
    if sum_linear > 1e-15:
        for c in components.values():
            c.percent_contribution = (c.linear_delta / sum_linear) * 100.0

    # 3. Compute Linear Taylor Bounds
    delta_linear_total = sum(c.linear_delta for c in components.values())
    lin_max = v_target_nom + delta_linear_total
    lin_min = v_target_nom - delta_linear_total

    # 4. Compute Statistical RSS Bounds
    delta_rss = math.sqrt(sum(c.linear_delta ** 2 for c in components.values()))
    rss_max = v_target_nom + delta_rss
    rss_min = v_target_nom - delta_rss

    # 5. Exact Extreme Value Analysis (EVA) via Non-Linear Corner Re-solves
    overrides_max = {c.name: c.param_wc_max for c in components.values() if c.tolerance > 0}
    overrides_min = {c.name: c.param_wc_min for c in components.values() if c.tolerance > 0}

    circ_max = _clone_circuit_with_overrides(circuit, overrides_max)
    circ_min = _clone_circuit_with_overrides(circuit, overrides_min)

    # Failed corner solves must propagate; a Taylor estimate is not a solved corner.
    exact_wc_max = float(solve(circ_max).V(target))
    exact_wc_min = float(solve(circ_min).V(target))

    # Ensure max >= min even in severe non-linear curvature
    if exact_wc_min > exact_wc_max:
        exact_wc_min, exact_wc_max = exact_wc_max, exact_wc_min

    return SensitivityResult(
        circuit=circuit,
        target_node=target,
        nominal_value=v_target_nom,
        components=components,
        worst_case_max=exact_wc_max,
        worst_case_min=exact_wc_min,
        linear_worst_case_max=lin_max,
        linear_worst_case_min=lin_min,
        rss_max=rss_max,
        rss_min=rss_min,
    )


def worst_case_analysis(
    circuit: Circuit,
    target_node: str,
    **kwargs,
) -> SensitivityResult:
    """Convenience alias for analyze_sensitivity."""
    return analyze_sensitivity(circuit, target_node, **kwargs)
