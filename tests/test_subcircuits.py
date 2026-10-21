"""
test_subcircuits.py — Comprehensive validation for SPICE subcircuits (.subckt / .ends),
hierarchical expansion, parameter evaluation, and pedagogical white-box inspection.
"""

import math
from pathlib import Path
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.engine.mna import solve
from circuit_solver.io.parser import parse_netlist_with_hierarchy
from circuit_solver.cli import main


def test_basic_subcircuit_expansion():
    """Verify standard SPICE .SUBCKT definition and X instance expansion."""
    netlist = """
    Test Basic Subcircuit
    .SUBCKT VOLTAGE_DIVIDER IN OUT GND
    R1 IN OUT 10k
    R2 OUT GND 10k
    .ENDS

    V1 1 0 DC 10
    X1 1 2 0 VOLTAGE_DIVIDER
    .END
    """
    elements, hierarchy = parse_netlist_with_hierarchy(netlist)
    assert len(elements) == 3  # V1, X1.R1, X1.R2
    circuit = Circuit(elements, hierarchy=hierarchy)
    sol = solve(circuit)

    assert pytest.approx(sol.V("1"), abs=1e-5) == 10.0
    assert pytest.approx(sol.V("2"), abs=1e-5) == 5.0


def test_subcircuit_with_parameters_and_expressions():
    """Verify PARAMS: defaults, instance overrides, and mathematical formulas."""
    netlist = """
    Test Parametric Subcircuit
    .SUBCKT ATTENUATOR IN OUT GND PARAMS: RA=10k RB=10k
    Rtop IN OUT {RA}
    Rbot OUT GND {RB}
    .ENDS

    V1 1 0 DC 12
    * Override RA=30k, RB=10k -> V(2) should be 12 * (10 / 40) = 3.0 V
    X1 1 2 0 ATTENUATOR PARAMS: RA=30k RB=10k
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    sol = solve(circuit)

    assert pytest.approx(sol.V("2"), abs=1e-5) == 3.0
    inst = circuit.hierarchy.find_instance("X1")
    assert inst is not None
    assert pytest.approx(inst.resolved_params["RA"]) == 30000.0
    assert pytest.approx(inst.resolved_params["RB"]) == 10000.0


def test_subcircuit_formula_evaluation():
    """Verify engineering formulas like {1 / (2 * pi * FC * C)} in subcircuits."""
    netlist = """
    Test Formula Evaluation
    .SUBCKT RC_CALC IN OUT PARAMS: FC=1000 C=100n
    R1 IN OUT {1 / (2 * pi * FC * C)}
    C1 OUT 0 {C}
    .ENDS

    V1 1 0 DC 5
    X1 1 2 RC_CALC PARAMS: FC=1000 C=100n
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    inst = circuit.hierarchy.find_instance("X1")
    assert inst is not None
    expected_r = 1.0 / (2.0 * math.pi * 1000.0 * 100e-9)
    assert pytest.approx(inst.resolved_params["FC"]) == 1000.0
    r_elem = next(el for el in circuit.elements if el.name == "X1.R1")
    assert pytest.approx(r_elem.resistance, rel=1e-4) == expected_r


def test_nested_subcircuits():
    """Verify hierarchical nested subcircuits (subcircuit instantiating another subcircuit)."""
    netlist = """
    Test Nested Subcircuits
    .SUBCKT HALF_DIVIDER IN OUT
    R_half IN OUT 5k
    .ENDS

    .SUBCKT FULL_DIVIDER IN OUT GND
    X_top IN OUT HALF_DIVIDER
    X_bot OUT GND HALF_DIVIDER
    .ENDS

    V1 1 0 DC 20
    X_main 1 2 0 FULL_DIVIDER
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    sol = solve(circuit)

    assert pytest.approx(sol.V("1"), abs=1e-5) == 20.0
    assert pytest.approx(sol.V("2"), abs=1e-5) == 10.0

    # Check hierarchy nesting
    inst_main = circuit.hierarchy.find_instance("X_MAIN")
    assert inst_main is not None
    assert len(inst_main.child_instances) == 2


def test_builtin_opamp_741_inverting_amplifier():
    """Verify built-in OPAMP_741 realistic macromodel in an inverting amplifier configuration.
    Gain = -Rf / Rin = -100k / 10k = -10.
    With Vin = 1.0 V, Vout should be very close to -10.0 V (accounting for finite AOL = 200k).
    """
    netlist = """
    Inverting Amplifier with Built-in 741
    X1 0 n_inv out OPAMP_741
    Rin in n_inv 10k
    Rf n_inv out 100k
    Vin in 0 DC 1.0
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    sol = solve(circuit)

    # Ideal gain is -10. With AOL=200k, finite gain error is < 0.01%
    assert pytest.approx(sol.V("out"), rel=1e-3) == -10.0
    # Virtual ground at inverting node
    assert pytest.approx(sol.V("n_inv"), abs=1e-4) == 0.0

    # Check that we can probe internal node of 741
    v_int = sol.V("X1.int_out")
    assert pytest.approx(v_int, rel=1e-3) == -10.0


def test_builtin_cmos_inverter():
    """Verify built-in CMOS_INVERTER macromodel operating with VDD=5V."""
    netlist = """
    CMOS Inverter with Built-in Model
    Vdd vdd 0 DC 5.0
    Vin in 0 DC 5.0
    Xinv in out vdd 0 CMOS_INVERTER
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    sol = solve(circuit)

    # When Vin = 5V (HIGH), NMOS conducts to GND and PMOS is OFF -> Vout ≈ 0 V
    assert sol.V("out") < 0.1


def test_pedagogical_white_box_inspection_and_summary():
    """Verify explain_hierarchy, hierarchy_tree, and subcircuit_summary reports."""
    netlist = """
    Pedagogical Subcircuit Demonstration
    .SUBCKT STAGE_AMP IN OUT GND PARAMS: GAIN=50
    Rin IN GND 1MEG
    E1 OUT GND IN GND {GAIN}
    .ENDS

    Vsig 1 0 DC 0.1
    Xamp 1 2 0 STAGE_AMP PARAMS: GAIN=20
    Rload 2 0 1k
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    sol = solve(circuit)

    # Check hierarchy tree view
    tree = circuit.hierarchy_tree()
    assert "XAMP" in tree
    assert "STAGE_AMP" in tree

    # Check pedagogical explain report
    report = circuit.explain_hierarchy()
    assert "RELATÓRIO PEDAGÓGICO DE EXPANSÃO HIERÁRQUICA" in report
    assert "STAGE_AMP" in report
    assert "GAIN = 20" in report
    assert "IMPACTO DIMENSIONAL NA MATRIZ MNA" in report

    # Check solution subcircuit summary
    sub_summary = sol.subcircuit_summary("XAMP")
    assert "Subcircuit Scope Analysis: [XAMP]" in sub_summary
    assert "Pino 'IN'" in sub_summary


def test_include_directive(tmp_path: Path):
    """Verify .include directive loads external subcircuit files."""
    lib_file = tmp_path / "custom_models.lib"
    lib_file.write_text(
        """
        * Custom Subcircuit Library
        .SUBCKT PRECISION_DIVIDER IN OUT GND
        R1 IN OUT 100k
        R2 OUT GND 100k
        .ENDS
        """,
        encoding="utf-8",
    )

    cir_file = tmp_path / "main.cir"
    cir_file.write_text(
        f"""
        Test Include Directive
        .INCLUDE "{lib_file}"
        V1 1 0 DC 24
        X1 1 2 0 PRECISION_DIVIDER
        .END
        """,
        encoding="utf-8",
    )

    circuit = Circuit.from_netlist(cir_file)
    sol = solve(circuit)

    assert pytest.approx(sol.V("1"), abs=1e-5) == 24.0
    assert pytest.approx(sol.V("2"), abs=1e-5) == 12.0


def test_cli_explain_and_hierarchy_commands(tmp_path: Path, capsys):
    """Verify pycircuit explain and pycircuit hierarchy CLI commands."""
    cir_file = tmp_path / "cli_subckt.cir"
    cir_file.write_text(
        """
        CLI Subcircuit Test
        .SUBCKT BUFFER IN OUT
        E1 OUT 0 IN 0 1.0
        .ENDS
        V1 1 0 DC 3.3
        Xbuf 1 2 BUFFER
        Rload 2 0 10k
        .END
        """,
        encoding="utf-8",
    )

    # Test pycircuit explain
    ret = main(["explain", str(cir_file)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "RELATÓRIO PEDAGÓGICO DE EXPANSÃO HIERÁRQUICA" in captured.out
    assert "XBUF" in captured.out

    # Test pycircuit hierarchy
    ret = main(["hierarchy", str(cir_file)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "XBUF (BUFFER)" in captured.out

    # Test pycircuit solve
    ret = main(["solve", str(cir_file)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "V(     2) =     3.300000 V" in captured.out
