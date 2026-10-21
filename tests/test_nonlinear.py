"""
test_nonlinear.py — Unit tests and analytical benchmarks for non-linear solver.

Validates:
- PN junction diode forward and reverse bias (Shockley model).
- Zener diode reverse breakdown voltage regulation.
- Bipolar Junction Transistors (BJT) in active and saturation regions.
- MOSFET in saturation and linear regions.
- Netlist parser support for D, Z, Q, M components.
- Convergence to tolerance epsilon < 1e-6.
"""

import pytest

from circuit_solver import (
    Circuit,
    Resistor,
    VoltageSource,
    Diode,
    ZenerDiode,
    BJT,
    MOSFET,
    solve,
    solve_nonlinear,
)
from circuit_solver.io.parser import parse_netlist_string


def test_diode_forward_bias():
    """Diode under DC forward bias: Shockley equation V_D ≈ 0.69 V, I_D ≈ 4.3 mA."""
    # V1: 5V from 1 to 0
    # R1: 1k from 1 to 2
    # D1: from 2 to 0
    c = Circuit([
        VoltageSource("V1", "1", "0", 5.0),
        Resistor("R1", "1", "2", 1000.0),
        Diode("D1", "2", "0", Is=1e-14, n=1.0, Vt=0.02585),
    ])

    sol = solve(c)
    assert c.has_nonlinear

    v2 = sol.V("2")
    # Expected: V(2) ~ 0.6925 V
    assert 0.65 < v2 < 0.72
    id_calc = sol.I_diode("D1")
    id_expected = (5.0 - v2) / 1000.0
    assert pytest.approx(id_calc, rel=1e-5) == id_expected
    assert sol.iterations <= 15


def test_diode_reverse_bias():
    """Diode under reverse bias should block current (I_D ≈ -Is)."""
    # V1: 5V from 0 to 1 -> V(1) = -5V
    # R1: 1k from 1 to 2
    # D1: anode at 2, cathode at 0 (reverse biased with ~ -5V)
    c = Circuit([
        VoltageSource("V1", "0", "1", 5.0),
        Resistor("R1", "1", "2", 1000.0),
        Diode("D1", "2", "0", Is=1e-14, n=1.0),
    ])

    sol = solve(c)
    assert pytest.approx(sol.V("1"), abs=1e-6) == -5.0
    # Node 2 should be virtually at -5V because diode is OFF
    assert pytest.approx(sol.V("2"), abs=1e-3) == -5.0
    assert abs(sol.I_diode("D1")) < 1e-11


def test_zener_voltage_regulator():
    """Zener diode in reverse breakdown acts as a voltage regulator (V_out ≈ Vz + I*Rz)."""
    # Supply: 12V
    # R_series: 1k
    # Zener: Vz = 5.1V, Rz = 10 ohms, Anode at 0, Cathode at 2
    c = Circuit([
        VoltageSource("V_IN", "1", "0", 12.0),
        Resistor("R_S", "1", "2", 1000.0),
        ZenerDiode("Z1", "0", "2", Vz=5.1, Rz=10.0),
    ])

    sol = solve(c)
    v_out = sol.V("2")
    # Analytical: (12 - 5.1)/(1000 + 10) = 6.9/1010 = 6.8317 mA
    # V_out = 5.1 + 0.0068317 * 10 = 5.1683 V
    assert pytest.approx(v_out, rel=1e-3) == 5.1683
    assert sol.iterations <= 10


def test_bjt_active_region():
    """NPN BJT common-emitter bias in active region (IC = beta * IB)."""
    # VCC: 10V
    # RC: 2k from 'cc' to 'c'
    # RB: 100k from 'in' to 'b'
    # VIN: 2V
    # Emitter: '0'
    c = Circuit([
        VoltageSource("VCC", "cc", "0", 10.0),
        VoltageSource("VIN", "in", "0", 2.0),
        Resistor("RC", "cc", "c", 2000.0),
        Resistor("RB", "in", "b", 100000.0),
        BJT("Q1", "c", "b", "0", polarity="NPN", beta=100.0, Is=1e-14),
    ])

    sol = solve(c)
    vb = sol.V("b")
    vc = sol.V("c")

    # V_BE ~ 0.65 - 0.72V
    assert 0.65 < vb < 0.72
    ib = (2.0 - vb) / 100000.0
    ic = (10.0 - vc) / 2000.0
    # In active region, IC / IB ≈ beta (100)
    assert pytest.approx(ic / ib, rel=0.05) == 100.0
    # V_CE > 0.2V (active mode)
    assert vc > 1.0


def test_bjt_saturation_switch():
    """NPN BJT driven into hard saturation (V_CE ≈ V_CE,sat ~ 0.07 V)."""
    # VCC = 5V, RC = 1k
    # VIN = 5V, RB = 10k
    c = Circuit([
        VoltageSource("VCC", "cc", "0", 5.0),
        VoltageSource("VIN", "in", "0", 5.0),
        Resistor("RC", "cc", "c", 1000.0),
        Resistor("RB", "in", "b", 10000.0),
        BJT("Q1", "c", "b", "0", polarity="NPN", beta=100.0, Is=1e-14),
    ])

    sol = solve(c)
    vc = sol.V("c")
    vb = sol.V("b")
    assert 0.65 < vb < 0.75
    # Saturated collector voltage
    assert 0.03 < vc < 0.15
    currents = sol.I_bjt("Q1")
    assert pytest.approx(currents["C"], rel=1e-2) == (5.0 - vc) / 1000.0


def test_mosfet_saturation():
    """NMOS in saturation: ID = 0.5 * Kp * (V_GS - V_th)^2 * (1 + lambda * V_DS)."""
    # VDD = 5V, RD = 2k
    # VG = 2.5V, S = 0
    # Vth = 1.0, Kp = 1e-3, lambda = 0.01
    c = Circuit([
        VoltageSource("VDD", "dd", "0", 5.0),
        VoltageSource("VG", "g", "0", 2.5),
        Resistor("RD", "dd", "d", 2000.0),
        MOSFET("M1", "d", "g", "0", polarity="NMOS", Vth=1.0, Kp=1e-3, lambda_param=0.01),
    ])

    sol = solve(c)
    vd = sol.V("d")
    # Analytical: Vd ≈ 2.6895 V
    assert pytest.approx(vd, rel=1e-3) == 2.6895
    assert sol.V_diff("d", "0") > (2.5 - 1.0)  # V_DS > V_GS - V_th -> saturation


def test_mosfet_linear_triode():
    """NMOS with heavy gate overdrive pulls drain low into triode region."""
    # VDD = 5V, RD = 10k
    # VG = 5V, Vth = 1.0 -> Vov = 4.0V
    # Pulls Vd very low (triode)
    c = Circuit([
        VoltageSource("VDD", "dd", "0", 5.0),
        VoltageSource("VG", "g", "0", 5.0),
        Resistor("RD", "dd", "d", 10000.0),
        MOSFET("M1", "d", "g", "0", polarity="NMOS", Vth=1.0, Kp=1e-3),
    ])

    sol = solve(c)
    vd = sol.V("d")
    assert vd < 0.2  # V_DS << V_GS - V_th -> triode


def test_netlist_parser_nonlinear():
    """Netlist reader correctly parses D, Z, Q, M components."""
    netlist = """Nonlinear test circuit
V1 1 0 10.0
R1 1 2 1k
D1 2 0
Z1 0 3 5.1 RZ=10
Q1 4 2 0 NPN BETA=120
M1 5 2 0 NMOS VTH=1.2 KP=2m
.end"""
    elements = parse_netlist_string(netlist)
    types = [type(el).__name__ for el in elements]
    assert "Diode" in types
    assert "ZenerDiode" in types
    assert "BJT" in types
    assert "MOSFET" in types

    # Check parsed attributes
    q = next(el for el in elements if isinstance(el, BJT))
    assert q.beta == 120.0
    assert q.polarity == "NPN"

    m = next(el for el in elements if isinstance(el, MOSFET))
    assert m.Vth == 1.2
    assert m.Kp == 2e-3


def test_tolerance_epsilon_convergence():
    """Residue convergence strictly satisfies epsilon < 1e-6."""
    c = Circuit([
        VoltageSource("V1", "1", "0", 3.3),
        Resistor("R1", "1", "2", 330.0),
        Diode("D1", "2", "0"),
    ])
    sol = solve_nonlinear(c, tol=1e-6)
    assert sol.iterations >= 2
    # Diode equation residual: I_R - I_D ≈ 0
    v2 = sol.V("2")
    ir = (3.3 - v2) / 330.0
    id_d = sol.I_diode("D1")
    assert abs(ir - id_d) < 1e-6


def test_pnp_bjt():
    """PNP BJT common-emitter circuit."""
    # Emitter tied to 5V supply
    # Base tied to ground via RB = 100k
    # Collector tied to ground via RC = 2k
    c = Circuit([
        VoltageSource("VEE", "e", "0", 5.0),
        Resistor("RB", "b", "0", 100000.0),
        Resistor("RC", "c", "0", 2000.0),
        BJT("Q1", "c", "b", "e", polarity="PNP", beta=100.0),
    ])
    sol = solve(c)
    ve = sol.V("e")
    vb = sol.V("b")
    vc = sol.V("c")
    assert ve == 5.0
    # V_EB ~ 0.65 - 0.72V -> V_B ~ 4.3V
    assert 4.2 < vb < 4.45
    # Collector pulled up by IC
    assert vc > 0.5


def test_pmos_saturation():
    """PMOS transistor in saturation."""
    # Source at 5V, Gate at 2.5V -> VSG = 2.5V > |Vth| = 1.0V (Vov = 1.5V)
    # Drain connected to ground through RD = 2k
    c = Circuit([
        VoltageSource("VDD", "s", "0", 5.0),
        VoltageSource("VG", "g", "0", 2.5),
        Resistor("RD", "d", "0", 2000.0),
        MOSFET("M1", "d", "g", "s", polarity="PMOS", Vth=-1.0, Kp=1e-3),
    ])
    sol = solve(c)
    vd = sol.V("d")
    # Current pulls Vd up to ~ 2.25V (VSD = 5 - 2.25 = 2.75V > Vov = 1.5V -> saturation)
    assert 2.0 < vd < 2.5
    currents = sol.I_mosfet("M1")
    assert pytest.approx(currents["S"], rel=1e-2) == (vd / 2000.0)
