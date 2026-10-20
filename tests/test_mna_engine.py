"""
test_mna_engine.py — Testes analíticos do motor MNA:
Divisor de tensão, fontes controladas (VCVS, VCCS), Op-Amps, auditoria de Tellegen e SPICE Parser.
"""

import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    CurrentSource,
    OpAmp,
    Resistor,
    VCCS,
    VCVS,
    VoltageSource,
)
from circuit_solver.engine.mna import solve
from circuit_solver.io.parser import parse_netlist_string


def test_voltage_divider():
    """Divisor resistivo simples: Vin = 10V, R1 = 1k, R2 = 1k -> V(mid) = 5V"""
    elements = [
        VoltageSource("V1", "in", "0", 10.0),
        Resistor("R1", "in", "mid", 1000.0),
        Resistor("R2", "mid", "0", 1000.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("in") == pytest.approx(10.0)
    assert sol.V("mid") == pytest.approx(5.0)
    assert sol.I_resistor("R1") == pytest.approx(5e-3)
    assert sol.I_resistor("R2") == pytest.approx(5e-3)
    assert sol.I_vsource("V1") == pytest.approx(-5e-3)

    # Teorema de Tellegen: soma das potências deve ser zero
    audit = sol.power_audit()
    assert audit["_total"] == pytest.approx(0.0, abs=1e-9)


def test_current_source_circuit():
    """Fonte de corrente I = 2A em paralelo com R = 10Ω -> V = 20V"""
    elements = [
        CurrentSource("I1", "1", "0", 2.0),
        Resistor("R1", "1", "0", 10.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(20.0)
    assert sol.I_resistor("R1") == pytest.approx(2.0)
    audit = sol.power_audit()
    assert audit["_total"] == pytest.approx(0.0, abs=1e-9)


def test_vcvs_controlled_source():
    """Fonte de tensão controlada por tensão (VCVS - E):
    Vin = 2V. VCVS tem ganho mu = 3 controlado por Vin.
    V_out deve ser 6V.
    """
    elements = [
        VoltageSource("Vin", "in", "0", 2.0),
        Resistor("Rin", "in", "0", 1000.0),
        VCVS("E1", "out", "0", "in", "0", 3.0),
        Resistor("Rload", "out", "0", 500.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("in") == pytest.approx(2.0)
    assert sol.V("out") == pytest.approx(6.0)
    assert sol.I_resistor("Rload") == pytest.approx(6.0 / 500.0)


def test_vccs_controlled_source():
    """Fonte de corrente controlada por tensão (VCCS - G):
    Vin = 4V. VCCS tem gm = 0.5 S controlado por Vin.
    I_out = 0.5 * 4 = 2A injetado em Rload = 10Ω -> V_out = 20V.
    """
    elements = [
        VoltageSource("Vin", "in", "0", 4.0),
        Resistor("Rin", "in", "0", 1000.0),
        VCCS("G1", "out", "0", "in", "0", 0.5),
        Resistor("Rload", "out", "0", 10.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("in") == pytest.approx(4.0)
    assert sol.V("out") == pytest.approx(20.0)


def test_ideal_opamp_inverting():
    """Amplificador operacional ideal inversor:
    Vin = 1V, R1 = 10k, Rf = 20k -> Vout = -2V
    """
    elements = [
        VoltageSource("Vin", "in", "0", 1.0),
        Resistor("R1", "in", "vn", 10000.0),
        Resistor("Rf", "vn", "out", 20000.0),
        Resistor("Rl", "out", "0", 5000.0),
        OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("in") == pytest.approx(1.0)
    assert sol.V("vn") == pytest.approx(0.0, abs=1e-12)  # Terra virtual
    assert sol.V("out") == pytest.approx(-2.0)


def test_spice_parser_integration(tmp_path):
    """Leitura e resolução a partir de netlist SPICE textual e arquivo."""
    netlist = """* Circuito de teste SPICE
V1 1 0 DC 12
R1 1 2 2k
R2 2 0 4k
E1 3 0 2 0 2.0
Rload 3 0 1k
G1 4 0 2 0 0.001
R3 4 0 1k
A1 0 5 6 0
R4 1 5 10k
R5 5 6 10k
.end
"""
    elements = parse_netlist_string(netlist)
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(12.0)
    assert sol.V("2") == pytest.approx(8.0)
    assert sol.V("3") == pytest.approx(16.0)

    # Test parse from file
    p = tmp_path / "test.cir"
    p.write_text(netlist)
    from circuit_solver.io.parser import parse_netlist
    file_elements = parse_netlist(p)
    assert len(file_elements) == len(elements)


def test_solution_methods_and_summary():
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "2", 100.0),
        Resistor("R2", "2", "0", 100.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert "V1" in repr(circuit) or "Circuit" in repr(circuit)
    assert "V(1)" in repr(sol)
    summary_text = sol.summary()
    assert "MNA Solution" in summary_text
    assert sol.V_diff("1", "2") == pytest.approx(5.0)
    assert sol.P_resistor("R1") == pytest.approx(0.25)
    assert sol.P_vsource("V1") == pytest.approx(0.5)


def test_monte_carlo_and_plots():
    import matplotlib
    matplotlib.use("Agg")
    from circuit_solver.engine.monte_carlo import monte_carlo
    from circuit_solver.visualization.plots import (
        plot_monte_carlo,
        plot_monte_carlo_all,
        plot_power_transfer,
    )
    from circuit_solver.engine.thevenin import thevenin_norton

    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "2", 1000.0, tolerance=0.05),
        Resistor("R2", "2", "0", 1000.0, tolerance=0.05),
    ]
    circuit = Circuit(elements)

    mc = monte_carlo(circuit, n_samples=50, seed=123)
    mc_summary = mc.summary()
    assert "Monte Carlo Analysis" in mc_summary
    assert mc.n_failed == 0
    assert len(mc.node_labels) == 2

    # Test plotting
    fig1 = plot_monte_carlo(mc, "2", show=False)
    assert fig1 is not None

    fig2 = plot_monte_carlo_all(mc, show=False)
    assert fig2 is not None

    th = thevenin_norton(circuit, "2", "0")
    assert "Thévenin / Norton Equivalent" in th.summary()
    fig3 = plot_power_transfer(th, show=False)
    assert fig3 is not None
