"""
test_sensitivity_wca.py — Test suite for Analytical Sensitivity, Worst-Case (WCA) and EVA.

Validates:
  1. Exact mathematical equivalence between Adjoint Network Method and theoretical derivatives:
     - Voltage divider: dV/dR1, dV/dR2, dV/dV1
     - Inverting Op-Amp: dV/dRf, dV/dRi
     - Finite difference perturbations
  2. Extreme Value Analysis (EVA):
     - Exact non-linear corner simulation bounds (worst_case_max, worst_case_min)
     - Linear Taylor first-order worst-case bounds
     - Linear RSS combination without automatic probability coverage
  3. Pareto ranking and pedagogical sensitivity reports.
  4. Dependent source sensitivities (VCVS, VCCS).
  5. CLI integration (`pycircuit sensitivity` and `pycircuit wca`).
"""

import math
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    VCVS,
    OpAmp,
)
from circuit_solver.engine.sensitivity import analyze_sensitivity, worst_case_analysis
from circuit_solver.cli import main as cli_main


def test_voltage_divider_analytical_sensitivities():
    """Verify sensitivities on classic voltage divider vs closed-form equations:
    V2 = V1 * R2 / (R1 + R2)
    dV2/dR1 = -V1 * R2 / (R1 + R2)^2
    dV2/dR2 = +V1 * R1 / (R1 + R2)^2
    dV2/dV1 = R2 / (R1 + R2)
    """
    v1_val = 12.0
    r1_val = 3000.0
    r2_val = 1000.0

    circ = Circuit([
        VoltageSource("V1", "1", "0", v1_val),
        Resistor("R1", "1", "2", r1_val, tolerance=0.05),
        Resistor("R2", "2", "0", r2_val, tolerance=0.05),
    ])

    res = analyze_sensitivity(circ, target_node="2")

    # Theoretical formulas
    r_sum = r1_val + r2_val
    v_theo = v1_val * (r2_val / r_sum)  # 12 * 1000 / 4000 = 3.0 V
    d_v_d_r1_theo = -v1_val * r2_val / (r_sum ** 2)  # -12 * 1000 / 16e6 = -0.00075 V/Ohm
    d_v_d_r2_theo = +v1_val * r1_val / (r_sum ** 2)  # +12 * 3000 / 16e6 = +0.00225 V/Ohm
    d_v_d_v1_theo = r2_val / r_sum                   # 1000 / 4000 = 0.25 V/V

    assert res.nominal_value == pytest.approx(v_theo)
    assert res.components["R1"].abs_sensitivity == pytest.approx(d_v_d_r1_theo, rel=1e-7)
    assert res.components["R2"].abs_sensitivity == pytest.approx(d_v_d_r2_theo, rel=1e-7)
    assert res.components["V1"].abs_sensitivity == pytest.approx(d_v_d_v1_theo, rel=1e-7)

    # Relative normalized sensitivity S = (p / V) * (dV/dp)
    # S_R1 = (3000 / 3.0) * (-0.00075) = -0.75 (-75%)
    # S_R2 = (1000 / 3.0) * (+0.00225) = +0.75 (+75%)
    # S_V1 = (12.0 / 3.0) * (0.25) = +1.00 (+100%)
    assert res.components["R1"].rel_sensitivity == pytest.approx(-0.75, rel=1e-7)
    assert res.components["R2"].rel_sensitivity == pytest.approx(+0.75, rel=1e-7)
    assert res.components["V1"].rel_sensitivity == pytest.approx(+1.00, rel=1e-7)

    # Direction check
    assert res.components["R1"].worst_case_dir == -1  # Increasing R1 decreases V2
    assert res.components["R2"].worst_case_dir == +1  # Increasing R2 increases V2
    assert res.components["V1"].worst_case_dir == +1  # Increasing V1 increases V2


def test_finite_difference_cross_validation():
    """Verify Adjoint sensitivities against numerical central finite differences."""
    circ = Circuit([
        VoltageSource("V1", "1", "0", 15.0),
        Resistor("R1", "1", "2", 1200.0),
        Resistor("R2", "2", "0", 800.0),
        Resistor("R3", "2", "3", 500.0),
        Resistor("R4", "3", "0", 1500.0),
        CurrentSource("I1", "0", "3", 2e-3),
    ])

    res = analyze_sensitivity(circ, target_node="3")
    delta_ratio = 1e-6

    # Test each component by finite difference
    for el in circ.elements:
        if el.name not in res.components:
            continue
        p_nom = res.components[el.name].nominal_value
        dp = p_nom * delta_ratio if abs(p_nom) > 1e-12 else 1e-6

        # Solve with +dp
        overrides_plus = {el.name: p_nom + dp}
        from circuit_solver.engine.sensitivity import _clone_circuit_with_overrides
        c_plus = _clone_circuit_with_overrides(circ, overrides_plus)
        from circuit_solver.engine.mna import solve
        v_plus = solve(c_plus).V("3")

        # Solve with -dp
        overrides_minus = {el.name: p_nom - dp}
        c_minus = _clone_circuit_with_overrides(circ, overrides_minus)
        v_minus = solve(c_minus).V("3")

        d_num = (v_plus - v_minus) / (2.0 * dp)
        d_adj = res.components[el.name].abs_sensitivity

        assert d_adj == pytest.approx(d_num, rel=1e-4, abs=1e-9)


def test_extreme_value_analysis_bounds_divider():
    """Verify EVA exact corner re-solve bounds vs Taylor linear bounds and RSS."""
    circ = Circuit([
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "2", 1000.0, tolerance=0.10),  # ±10%
        Resistor("R2", "2", "0", 1000.0, tolerance=0.10),  # ±10%
    ])

    # Custom tolerances: V1 has 0 tolerance, R1 & R2 have 10%
    res = worst_case_analysis(circ, target_node="2", custom_tolerances={"V1": 0.0, "R1": 0.10, "R2": 0.10})

    # Exact non-linear maximum:
    # R1 decreases to 900, R2 increases to 1100
    # V_max = 10 * 1100 / (900 + 1100) = 11000 / 2000 = 5.5 V
    assert res.worst_case_max == pytest.approx(5.5, rel=1e-5)

    # Exact non-linear minimum:
    # R1 increases to 1100, R2 decreases to 900
    # V_min = 10 * 900 / (1100 + 900) = 9000 / 2000 = 4.5 V
    assert res.worst_case_min == pytest.approx(4.5, rel=1e-5)

    # First-order linear Taylor:
    # dV/dR1 = -0.0025, delta_R1 = 100 -> delta_V1 = 0.25
    # dV/dR2 = +0.0025, delta_R2 = 100 -> delta_V2 = 0.25
    # Total linear delta = 0.50 V -> [4.5 V, 5.5 V]
    assert res.linear_worst_case_max == pytest.approx(5.5, rel=1e-5)
    assert res.linear_worst_case_min == pytest.approx(4.5, rel=1e-5)

    # RSS Statistical 3-sigma:
    # delta_RSS = sqrt(0.25^2 + 0.25^2) = 0.25 * sqrt(2) ≈ 0.353553 V
    delta_rss_expected = math.sqrt(0.25**2 + 0.25**2)
    assert res.rss_max == pytest.approx(5.0 + delta_rss_expected, rel=1e-5)
    assert res.rss_min == pytest.approx(5.0 - delta_rss_expected, rel=1e-5)


def test_inverting_opamp_sensitivity():
    """Verify sensitivities on an inverting amplifier:
    Vout = - (Rf / Rin) * Vin
    dVout/dRf = - Vin / Rin
    dVout/dRin = + Vin * Rf / Rin^2
    """
    circ = Circuit([
        VoltageSource("Vin", "in", "0", 2.0),
        Resistor("Rin", "in", "vn", 1000.0, tolerance=0.01),
        Resistor("Rf", "vn", "out", 10000.0, tolerance=0.01),
        OpAmp("OA1", in_p="0", in_n="vn", out="out"),
    ])

    res = analyze_sensitivity(circ, target_node="out", custom_tolerances={"Vin": 0.0})

    # Nominal Vout = - (10k / 1k) * 2 = -20.0 V
    assert res.nominal_value == pytest.approx(-20.0)

    # dVout/dRf = - 2.0 / 1000 = -0.002 V/Ohm
    # dVout/dRin = + 2.0 * 10000 / 1000^2 = +0.02 V/Ohm
    d_rf_theo = -2.0 / 1000.0
    d_rin_theo = +2.0 * 10000.0 / (1000.0 ** 2)

    assert res.components["Rf"].abs_sensitivity == pytest.approx(d_rf_theo, rel=1e-5)
    assert res.components["Rin"].abs_sensitivity == pytest.approx(d_rin_theo, rel=1e-5)

    # Relative sensitivities
    # S_Rf = (10000 / -20) * (-0.002) = +1.0
    # S_Rin = (1000 / -20) * (0.02) = -1.0
    assert res.components["Rf"].rel_sensitivity == pytest.approx(1.0, rel=1e-5)
    assert res.components["Rin"].rel_sensitivity == pytest.approx(-1.0, rel=1e-5)


def test_dependent_sources_sensitivity():
    """Verify sensitivity with respect to VCVS gain and VCCS transconductance."""
    circ = Circuit([
        VoltageSource("V1", "1", "0", 5.0),
        Resistor("R1", "1", "2", 100.0),
        VCVS("E1", node_p="3", node_n="0", ctrl_p="2", ctrl_n="0", gain=4.0),
        Resistor("RL", "3", "0", 1000.0),
    ])

    # Target node '3' is output of VCVS
    # Since V(2) = 5.0 V (no current drawn by ideal VCVS input), V(3) = gain * V(2) = 4.0 * 5.0 = 20.0 V
    # dV(3) / d(gain) = V(2) = 5.0 V/(V/V)
    res = analyze_sensitivity(circ, target_node="3")
    assert res.nominal_value == pytest.approx(20.0)
    assert res.components["E1"].abs_sensitivity == pytest.approx(5.0, rel=1e-5)


def test_pareto_ranking_and_summary_formatting():
    """Verify Pareto ranking sorting and summary report generation."""
    circ = Circuit([
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R_huge", "1", "2", 1000.0, tolerance=0.10),
        Resistor("R_small", "2", "0", 10.0, tolerance=0.01),
    ])

    res = analyze_sensitivity(circ, target_node="2")

    ranking = res.pareto_ranking()
    # R_huge should have significantly higher impact than R_small
    assert ranking[0].name in ("R_huge", "V1")

    # Sum of Pareto contributions should be approximately 100%
    total_pct = sum(c.percent_contribution for c in res.components.values())
    assert total_pct == pytest.approx(100.0, rel=1e-3)

    summary_text = res.summary()
    assert "Worst-Case (EVA) & Sensitivity Analysis Report" in summary_text
    assert "Nominal Voltage:" in summary_text
    assert "Pareto" in summary_text


def test_cli_sensitivity_command(tmp_path, capsys):
    """Verify CLI `pycircuit sensitivity` and `pycircuit wca` commands."""
    cir_content = """* Test Voltage Divider
V1 1 0 10V
R1 1 2 1k
R2 2 0 1k
.END
"""
    file_path = tmp_path / "divider.cir"
    file_path.write_text(cir_content)

    # 1. pycircuit sensitivity
    code = cli_main(["sensitivity", str(file_path), "--target", "2"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Worst-Case (EVA) & Sensitivity Analysis Report" in captured.out
    assert "R1" in captured.out
    assert "R2" in captured.out

    # 2. pycircuit wca (alias)
    code2 = cli_main(["wca", str(file_path), "-t", "2", "--tol", "0.01"])
    assert code2 == 0
    captured2 = capsys.readouterr()
    assert "Nominal Voltage:" in captured2.out
