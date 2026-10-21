"""Corner coverage checked against closed-form interaction and feedback models."""
import numpy as np
import pytest

from circuit_solver import Circuit, Resistor, VoltageSource, VCVS, analyze_sensitivity
from circuit_solver.cli import main
from circuit_solver.engine.mna import solve
from circuit_solver.engine.sensitivity import _clone_circuit_with_overrides


def interaction_circuit():
    # Vout = (g1 - 1)(g2 - 1) V: both first derivatives vanish at (1, 1).
    return Circuit([
        VoltageSource("Vref", "ref", "0", 1),
        VCVS("E1", "a", "0", "ref", "0", 1),
        VoltageSource("Vshift", "a", "b", 1),
        VCVS("E2", "c", "0", "b", "0", 1),
        VCVS("Esub", "c", "out", "b", "0", 1),
        Resistor("Rload", "out", "0", 1000, tolerance=0),
    ])


def analyze_interaction(**kwargs):
    return analyze_sensitivity(
        interaction_circuit(), "out", custom_tolerances={"E1": .2, "E2": .3},
        default_source_tolerance=0, default_resistor_tolerance=0, **kwargs,
    )


def test_zero_nominal_derivatives_do_not_remove_interaction_corners():
    result = analyze_interaction()
    assert result.corner_method == "exhaustive"
    assert result.varying_parameters == 2
    assert result.corners_evaluated == 4
    assert result.worst_case_min == pytest.approx(-.06)
    assert result.worst_case_max == pytest.approx(.06)
    assert result.rss_min == pytest.approx(0)
    assert result.linear_worst_case_max == pytest.approx(0)
    for name in ("E1", "E2"):
        assert result.components[name].abs_sensitivity == pytest.approx(0)
    for params, expected in [(result.worst_case_min_parameters, -.06),
                             (result.worst_case_max_parameters, .06)]:
        assert solve(_clone_circuit_with_overrides(result.circuit, params)).V("out") == pytest.approx(expected)
    assert result.circuit.elements[1].gain == 1
    assert result.circuit.elements[3].gain == 1


def test_selected_method_and_automatic_budget_fallback_are_explicit():
    for kwargs in ({"corner_method": "selected"}, {"max_corner_parameters": 1}):
        result = analyze_interaction(**kwargs)
        assert result.corner_method == "selected"
        assert result.corners_evaluated == 2
        assert result.worst_case_min == pytest.approx(0)
        assert result.worst_case_max == pytest.approx(0)
        assert "no global bound" in result.summary()
    with pytest.raises(ValueError, match="require 2 parameters"):
        analyze_interaction(corner_method="exhaustive", max_corner_parameters=1)


def test_corner_extrema_do_not_claim_to_bound_an_interior_pole():
    # Vout = g/(1-g); g in [-.25, 1.25] crosses a pole at 1.
    circuit = Circuit([
        VoltageSource("V1", "fb", "out", 1),
        VCVS("E1", "out", "0", "fb", "0", .5),
        Resistor("R1", "out", "0", 1000, tolerance=0),
    ])
    result = analyze_sensitivity(circuit, "out", custom_tolerances={"E1": 1.5},
                                 default_source_tolerance=0, default_resistor_tolerance=0)
    assert result.corners_evaluated == 2
    assert result.worst_case_min == pytest.approx(-5)
    assert result.worst_case_max == pytest.approx(-.2)
    assert result.nominal_value == pytest.approx(1)
    assert "poles or other extrema may exist" in result.summary()


def test_singular_corner_aborts_instead_of_reporting_partial_extrema():
    circuit = Circuit([
        VoltageSource("V1", "fb", "out", 1),
        VCVS("E1", "out", "0", "fb", "0", .5),
        Resistor("R1", "out", "0", 1000, tolerance=0),
    ])
    with pytest.raises(np.linalg.LinAlgError) as exc:
        analyze_sensitivity(circuit, "out", custom_tolerances={"E1": 1},
                            default_source_tolerance=0, default_resistor_tolerance=0)
    diagnostic = str(exc.value) + " ".join(getattr(exc.value, "__notes__", []))
    assert "corner failed" in diagnostic and "E1" in diagnostic


def test_fixed_and_zero_nominal_parameters_require_no_extra_solve():
    circuit = Circuit([VoltageSource("V1", "out", "0", 0), Resistor("R1", "out", "0", 1000, tolerance=0)])
    result = analyze_sensitivity(circuit, "out", default_resistor_tolerance=0)
    assert result.corners_evaluated == 0
    assert result.varying_parameters == 0
    assert result.worst_case_min == result.worst_case_max == 0


@pytest.mark.parametrize("kwargs", [{"corner_method": "unknown"}, {"max_corner_parameters": -1},
                                    {"max_corner_parameters": 1.5}, {"max_corner_parameters": True}])
def test_invalid_corner_options(kwargs):
    with pytest.raises(ValueError):
        analyze_interaction(**kwargs)


def test_cli_exposes_corner_method_and_budget(tmp_path, capsys):
    netlist = tmp_path / "divider.cir"
    netlist.write_text("Divider\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.end\n")
    assert main(["wca", str(netlist), "--target", "out", "--corners", "exhaustive",
                 "--max-corner-parameters", "3"]) == 0
    assert "exhaustive; solves: 8" in capsys.readouterr().out
