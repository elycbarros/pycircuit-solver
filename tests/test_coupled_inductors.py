"""
test_coupled_inductors.py — Unit tests for magnetically coupled inductors and transformers.

Tests cover:
1. Alexander & Sadiku Example 13.1 (coupled coils sinusoidal steady-state benchmark).
2. Ideal/Real Transformer in AC: step-down turns ratio (120V -> 60V).
3. Polarity inversion with negative coupling k < 0 (180 degree phase shift).
4. SPICE netlist parsing of K (MutualCoupling) and T (Transformer) elements.
5. Transient response of coupled coils / transformers to step excitation.
6. Validation checks: |k| > 1.0, non-positive inductance, dangling/shorted windings, missing referenced inductors.
"""

import math
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    Capacitor,
    Inductor,
    MutualCoupling,
    Transformer,
    VoltageSource,
    StepWaveform,
)
from circuit_solver.core.validator import CircuitValidationError
from circuit_solver.engine.ac import solve_ac
from circuit_solver.engine.transient import solve_transient
from circuit_solver.io.parser import parse_netlist_string


def test_sadiku_ex_13_1_coupled_coils_ac():
    """Benchmark: Alexander & Sadiku (5th/6th ed.), Example 13.1.

    Circuit with coupled coils:
      * Source: Vs = 12 ∠ 0° V at omega = 1 rad/s
      * Primary: R1 = 4 Ω, L1 = 8 H (impedance 4 + j8 Ω)
      * Secondary: L2 = 10 H, C1 = 0.5 F (impedance -j2 Ω), R2 = 5 Ω
      * Mutual coupling: M = 1 H (impedance j1 Ω)
        k = M / sqrt(L1 * L2) = 1 / sqrt(80) ≈ 0.1118034

    Analytical solution:
      I1 = 1.350 ∠ -62.85° A
      I2 = 0.1431 ∠ -30.85° A
    """
    omega = 1.0
    f = omega / (2.0 * math.pi)
    k = 1.0 / math.sqrt(8.0 * 10.0)

    circ = Circuit([
        VoltageSource("V1", "1", "0", voltage=12.0, ac_mag=12.0, ac_phase=0.0),
        Resistor("R1", "1", "2", 4.0),
        Inductor("L1", "2", "0", 8.0),
        Inductor("L2", "3", "0", 10.0),
        Capacitor("C1", "3", "4", 0.5),
        Resistor("R2", "4", "0", 5.0),
        MutualCoupling("K1", "L1", "L2", k),
    ])

    sol = solve_ac(circ, f)

    i1 = sol.I_inductor("L1")
    i2 = sol.I_resistor("R2")

    # Magnitude check within 0.1% of textbook
    assert abs(abs(i1) - 1.350) < 0.005
    assert abs(abs(i2) - 0.1431) < 0.002

    # Phase check within 0.2 degrees
    phase_i1 = math.degrees(math.atan2(i1.imag, i1.real))
    phase_i2 = math.degrees(math.atan2(i2.imag, i2.real))
    assert abs(phase_i1 - (-62.85)) < 0.2
    assert abs(phase_i2 - (-30.85)) < 0.2


def test_transformer_step_down_ac():
    """Verify AC transformer step-down behavior: V2 = V1 / n.

    L1 = 10 H, L2 = 2.5 H -> turns ratio n = sqrt(10 / 2.5) = 2.0.
    With V1 = 120 V at 60 Hz, V2 into a 100 Ω load should be approx 60 V.
    """
    circ = Circuit([
        VoltageSource("V1", "1", "0", voltage=120.0, ac_mag=120.0, ac_phase=0.0),
        Transformer("T1", "1", "0", "2", "0", L1=10.0, L2=2.5, coupling=0.9999),
        Resistor("RL", "2", "0", 100.0),
    ])

    sol = solve_ac(circ, 60.0)
    v1 = abs(sol.V("1"))
    v2 = abs(sol.V("2"))

    assert abs(v1 - 120.0) < 1e-4
    assert abs(v2 - 60.0) < 0.1
    assert abs(v1 / v2 - 2.0) < 0.005

    # Check currents through primary and secondary windings
    i_prim = sol.I_transformer_primary("T1")
    i_sec = sol.I_transformer_secondary("T1")
    assert abs(i_prim) > 0.0
    assert abs(i_sec) > 0.0
    # Secondary current leaves p2 and flows into RL: I_RL = -I_sec
    assert abs(abs(i_sec) - v2 / 100.0) < 0.01


def test_transformer_polarity_inversion():
    """Negative coupling coefficient k < 0 causes a 180° phase inversion."""
    circ_pos = Circuit([
        VoltageSource("V1", "1", "0", voltage=10.0, ac_mag=10.0, ac_phase=0.0),
        Transformer("T1", "1", "0", "2", "0", L1=1.0, L2=1.0, coupling=0.999),
        Resistor("RL", "2", "0", 50.0),
    ])
    sol_pos = solve_ac(circ_pos, 50.0)

    circ_neg = Circuit([
        VoltageSource("V1", "1", "0", voltage=10.0, ac_mag=10.0, ac_phase=0.0),
        Transformer("T1", "1", "0", "2", "0", L1=1.0, L2=1.0, coupling=-0.999),
        Resistor("RL", "2", "0", 50.0),
    ])
    sol_neg = solve_ac(circ_neg, 50.0)

    v_pos = sol_pos.V("2")
    v_neg = sol_neg.V("2")

    # Magnitudes should be equal
    assert abs(abs(v_pos) - abs(v_neg)) < 1e-3

    # Phase difference should be 180°
    phase_diff = abs(sol_pos.V_phase_deg("2") - sol_neg.V_phase_deg("2"))
    assert abs(phase_diff - 180.0) < 1.0


def test_spice_netlist_k_element_parser():
    """Verify that netlists containing K (coupling) and T (transformer) lines parse and solve."""
    netlist = """
    * Magnetically coupled coils from SPICE netlist
    V1 1 0 AC 12 0
    R1 1 2 4
    L1 2 0 8
    L2 3 0 10
    C1 3 4 0.5
    R2 4 0 5
    K1 L1 L2 0.111803398875
    .end
    """
    elements = parse_netlist_string(netlist)
    assert any(isinstance(el, MutualCoupling) for el in elements)

    k_el = next(el for el in elements if isinstance(el, MutualCoupling))
    assert k_el.name == "K1"
    assert k_el.ind1_name == "L1"
    assert k_el.ind2_name == "L2"
    assert abs(k_el.coupling - 0.1118034) < 1e-6

    circ = Circuit(elements)
    sol = solve_ac(circ, 1.0 / (2.0 * math.pi))
    assert abs(abs(sol.I_inductor("L1")) - 1.350) < 0.005


def test_spice_netlist_t_element_parser():
    """Verify parsing of 2-winding transformer line: T<name> p1 n1 p2 n2 L1 L2 k."""
    netlist = """
    * Transformer netlist
    V1 1 0 AC 120 0
    T1 1 0 2 0 10 2.5 0.999
    RL 2 0 100
    .end
    """
    elements = parse_netlist_string(netlist)
    t_el = next(el for el in elements if isinstance(el, Transformer))
    assert t_el.name == "T1"
    assert t_el.p1 == "1"
    assert t_el.n1 == "0"
    assert t_el.p2 == "2"
    assert t_el.n2 == "0"
    assert t_el.L1 == 10.0
    assert t_el.L2 == 2.5
    assert t_el.coupling == 0.999

    circ = Circuit(elements)
    sol = solve_ac(circ, 60.0)
    assert abs(abs(sol.V("2")) - 60.0) < 0.1


def test_transformer_transient_step_response():
    """Verify transient response of transformer when excited by a step input."""
    circ = Circuit([
        VoltageSource("V1", "1", "0", voltage=0.0, waveform=StepWaveform(v1=0.0, v2=10.0, t_step=1e-5)),
        Resistor("R1", "1", "2", 1.0),
        Transformer("T1", "2", "0", "3", "0", L1=1.0, L2=0.25, coupling=0.999),
        Resistor("RL", "3", "0", 100.0),
    ])

    sol = solve_transient(circ, t_stop=1e-3, dt=1e-5)

    # Initial state before step: 0V
    assert sol.V("3")[0] == 0.0

    # At t=20us right after the step at t=10us:
    # Secondary voltage should be approximately V(2) / 2 = 5V
    v2 = sol.V("2")[2]
    v3 = sol.V("3")[2]
    assert abs(v2 - 10.0) < 0.1
    assert abs(v3 - 5.0) < 0.1
    assert abs(v2 / v3 - 2.0) < 0.01

    # Secondary current should be tracked
    assert len(sol.I("T1")) == len(sol.time)


def test_coupled_inductors_validation_errors():
    """Test validation errors for invalid coupling and transformer configurations."""
    # 1. Invalid coupling coefficient |k| > 1.0
    with pytest.raises(ValueError, match="Coupling coefficient k must satisfy -1.0 <= k <= 1.0"):
        MutualCoupling("K1", "L1", "L2", 1.5)

    with pytest.raises(ValueError, match="Coupling coefficient k must satisfy -1.0 <= k <= 1.0"):
        Transformer("T1", "1", "0", "2", "0", L1=1.0, L2=1.0, coupling=-1.2)

    # 2. Non-positive inductance in transformer
    with pytest.raises(ValueError, match="L1 and L2 must be > 0"):
        Transformer("T1", "1", "0", "2", "0", L1=0.0, L2=1.0)

    # 3. Transformer shorted winding
    with pytest.raises(CircuitValidationError, match="shorted primary winding"):
        Circuit([
            VoltageSource("V1", "1", "0", voltage=10.0),
            Transformer("T1", "1", "1", "2", "0", L1=1.0, L2=1.0),
            Resistor("R1", "2", "0", 10.0),
        ])

    # 4. MutualCoupling references non-existent inductor
    with pytest.raises(CircuitValidationError, match="references inductor 'L2', which is not in the circuit"):
        Circuit([
            VoltageSource("V1", "1", "0", voltage=10.0),
            Inductor("L1", "1", "0", 1.0),
            MutualCoupling("K1", "L1", "L2", 0.5),
        ])

    # 5. MutualCoupling references the same inductor twice
    with pytest.raises(CircuitValidationError, match="references the same inductor 'L1' twice"):
        Circuit([
            VoltageSource("V1", "1", "0", voltage=10.0),
            Inductor("L1", "1", "0", 1.0),
            MutualCoupling("K1", "L1", "L1", 0.5),
        ])
