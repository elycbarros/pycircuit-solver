"""
test_transient.py — Analytical benchmark tests for time-domain transient analysis.

Validates:
1. RC series charging step response vs exact analytical formula: V_C(t) = V0*(1 - exp(-t/tau)).
2. RC capacitor discharge with initial condition (UIC): V_C(t) = V0*exp(-t/tau).
3. RL series inductor current step response: I_L(t) = (V0/R)*(1 - exp(-t/tau)).
4. RLC series second-order underdamped oscillation and overshoot.
5. SPICE PULSE and SIN waveforms parsing and transient evaluation.
6. CLI `pycircuit tran` invocation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    StepWaveform,
    PulseWaveform,
    SineWaveform,
)
from circuit_solver.engine.transient import solve_transient
from circuit_solver.io.parser import parse_netlist_string
from circuit_solver.cli import main as cli_main


# ---------------------------------------------------------------------------
# 1. RC Series Charging Step Response
# ---------------------------------------------------------------------------

def test_rc_charging_step_response_trapezoidal():
    """RC circuit with V=10V, R=1kΩ, C=1μF → tau = 1 ms.

    Exact voltage: V(t) = 10 * (1 - exp(-t / 1e-3)).
    At t = 1 ms (1 tau): V ≈ 6.321206 V.
    At t = 3 ms (3 tau): V ≈ 9.502129 V.
    At t = 5 ms (5 tau): V ≈ 9.932621 V.
    """
    R_val = 1000.0
    C_val = 1e-6
    tau = R_val * C_val

    # V1 starts at 0 and steps to 10V at t=0
    v_src = VoltageSource("V1", "1", "0", 10.0, waveform=StepWaveform(0.0, 10.0, 0.0))
    res = Resistor("R1", "1", "2", R_val)
    cap = Capacitor("C1", "2", "0", C_val, ic=0.0)

    circ = Circuit([v_src, res, cap])

    t_stop = 5.0 * tau
    dt = 1e-5  # 10 μs step

    sol = solve_transient(circ, t_stop=t_stop, dt=dt, method="trapezoidal", uic=True)

    v2_arr = sol.V("2")
    t_arr = sol.time

    # Evaluate at 1 tau, 3 tau, 5 tau
    for target_t, exact_v in [
        (1.0 * tau, 10.0 * (1.0 - math.exp(-1.0))),
        (3.0 * tau, 10.0 * (1.0 - math.exp(-3.0))),
        (5.0 * tau, 10.0 * (1.0 - math.exp(-5.0))),
    ]:
        idx = int(round(target_t / dt))
        sim_v = v2_arr[idx]
        assert abs(sim_v - exact_v) < 0.02, (
            f"At t={target_t*1000:.1f}ms: sim={sim_v:.4f}V, exact={exact_v:.4f}V"
        )


def test_rc_charging_backward_euler():
    """Test Backward Euler integration on RC step response."""
    R_val = 1000.0
    C_val = 1e-6
    tau = R_val * C_val

    v_src = VoltageSource("V1", "1", "0", 10.0, waveform=StepWaveform(0.0, 10.0, 0.0))
    res = Resistor("R1", "1", "2", R_val)
    cap = Capacitor("C1", "2", "0", C_val, ic=0.0)

    circ = Circuit([v_src, res, cap])

    t_stop = 3.0 * tau
    dt = 1e-5

    sol = solve_transient(circ, t_stop=t_stop, dt=dt, method="euler", uic=True)

    # At t = 1 tau
    idx = int(round(tau / dt))
    exact_v = 10.0 * (1.0 - math.exp(-1.0))
    sim_v = sol.V("2")[idx]

    # Euler is 1st-order, so slightly looser tolerance than trapezoidal
    assert abs(sim_v - exact_v) < 0.05, f"Euler at 1 tau: sim={sim_v}, exact={exact_v}"


# ---------------------------------------------------------------------------
# 2. Capacitor Discharge with Initial Condition (UIC)
# ---------------------------------------------------------------------------

def test_rc_discharge_uic():
    """Capacitor pre-charged to 8V discharging through 500Ω.

    Exact: V(t) = 8.0 * exp(-t / tau), tau = 500 * 2e-6 = 1 ms.
    At t = 1 ms: V ≈ 8 * exp(-1) = 2.9430 V.
    """
    R_val = 500.0
    C_val = 2e-6
    tau = R_val * C_val

    cap = Capacitor("C1", "1", "0", C_val, ic=8.0)
    res = Resistor("R1", "1", "0", R_val)

    circ = Circuit([cap, res])

    sol = solve_transient(circ, t_stop=3.0 * tau, dt=1e-5, method="trapezoidal", uic=True)

    idx_1ms = int(round(tau / 1e-5))
    expected = 8.0 * math.exp(-1.0)
    actual = sol.V("1")[idx_1ms]

    assert abs(actual - expected) < 0.05, f"Discharge at 1 tau: actual={actual}, expected={expected}"


# ---------------------------------------------------------------------------
# 3. RL Series Inductor Current Step Response
# ---------------------------------------------------------------------------

def test_rl_step_response_current():
    """RL circuit: Vs = 12V, R = 6Ω, L = 30mH.

    Time constant: tau = L / R = 0.030 / 6 = 5 ms.
    Steady-state current: I_max = Vs / R = 2.0 A.
    Exact current: I_L(t) = 2.0 * (1 - exp(-t / 5ms)).
    At t = 5 ms (1 tau): I_L = 2.0 * (1 - exp(-1)) ≈ 1.26424 A.
    """
    Vs = 12.0
    R_val = 6.0
    L_val = 0.030
    tau = L_val / R_val

    v_src = VoltageSource("V1", "1", "0", Vs, waveform=StepWaveform(0.0, Vs, 0.0))
    res = Resistor("R1", "1", "2", R_val)
    ind = Inductor("L1", "2", "0", L_val, ic=0.0)

    circ = Circuit([v_src, res, ind])

    sol = solve_transient(circ, t_stop=4.0 * tau, dt=1e-5, method="trapezoidal", uic=True)

    idx_tau = int(round(tau / 1e-5))
    expected_i = (Vs / R_val) * (1.0 - math.exp(-1.0))
    sim_i = sol.I("L1")[idx_tau]

    assert abs(sim_i - expected_i) < 0.01, f"RL current at 1 tau: sim={sim_i:.4f}A, expected={expected_i:.4f}A"


# ---------------------------------------------------------------------------
# 4. RLC Series Second-Order Underdamped Oscillation
# ---------------------------------------------------------------------------

def test_rlc_underdamped_step_response():
    """Series RLC: R = 20Ω, L = 10mH, C = 1μF, step V1 = 10V.

    omega0 = 1 / sqrt(LC) = 1 / sqrt(1e-5) = 10000 rad/s.
    alpha = R / (2L) = 20 / 0.02 = 1000 s^-1.
    omega_d = sqrt(omega0^2 - alpha^2) = sqrt(1e8 - 1e6) = sqrt(99e6) ≈ 9949.87 rad/s.

    Because alpha < omega0, the circuit must overshoot 10V.
    Peak time: tp = pi / omega_d ≈ 3.14159 / 9949.87 ≈ 0.3157 ms.
    Overshoot: V_peak = 10 * (1 + exp(-alpha * tp)) ≈ 10 * (1 + exp(-0.3157)) ≈ 17.29 V.
    """
    R_val = 20.0
    L_val = 10e-3
    C_val = 1e-6

    v_src = VoltageSource("V1", "1", "0", 10.0, waveform=StepWaveform(0.0, 10.0, 0.0))
    res = Resistor("R1", "1", "2", R_val)
    ind = Inductor("L1", "2", "3", L_val, ic=0.0)
    cap = Capacitor("C1", "3", "0", C_val, ic=0.0)

    circ = Circuit([v_src, res, ind, cap])

    sol = solve_transient(circ, t_stop=8e-3, dt=1e-6, method="trapezoidal", uic=True)

    v_c = sol.V("3")
    max_v = float(np.max(v_c))

    # Analytical overshoot is ~17.29 V
    assert max_v > 16.5 and max_v < 18.0, f"RLC peak voltage: {max_v:.2f} V, expected ≈ 17.29 V"

    # Final steady state should settle back to 10 V
    v_final = float(v_c[-1])
    assert abs(v_final - 10.0) < 0.2, f"RLC final voltage: {v_final:.2f} V, expected ≈ 10.0 V"


# ---------------------------------------------------------------------------
# 5. SPICE Netlist with PULSE and SIN Sources
# ---------------------------------------------------------------------------

def test_spice_pulse_netlist_transient():
    """Test parsing and simulation of standard SPICE PULSE waveform."""
    netlist = """
    * RC circuit with PULSE source
    Vpulse in 0 PULSE(0 5 0.5m 10u 10u 1m 3m)
    R1 in out 1k
    C1 out 0 1u
    .end
    """
    circuit = Circuit(parse_netlist_string(netlist))

    sol = solve_transient(circuit, t_stop=3e-3, dt=1e-5, method="trapezoidal", uic=True)

    # Before pulse starts (t = 0.2 ms), V(in) = 0
    idx_pre = int(round(0.2e-3 / 1e-5))
    assert sol.V("in")[idx_pre] == 0.0

    # During pulse high flat region (t = 1.0 ms), V(in) = 5
    idx_pulse = int(round(1.0e-3 / 1e-5))
    assert abs(sol.V("in")[idx_pulse] - 5.0) < 1e-6

    # Output V(out) should be charging towards 5V
    assert sol.V("out")[idx_pulse] > 1.8


def test_spice_sine_netlist_transient():
    """Test parsing and simulation of standard SPICE SIN waveform."""
    netlist = """
    * AC sine source into resistor
    Vsin in 0 SIN(0 10 1000 0 0)
    R1 in 0 100
    .end
    """
    circuit = Circuit(parse_netlist_string(netlist))

    sol = solve_transient(circuit, t_stop=1e-3, dt=1e-6, method="trapezoidal", uic=True)

    # At 1 kHz, period is 1 ms. Peak at 0.25 ms should be ≈ +10V
    idx_peak = int(round(0.25e-3 / 1e-6))
    assert abs(sol.V("in")[idx_peak] - 10.0) < 0.1


# ---------------------------------------------------------------------------
# 6. CLI `pycircuit tran` Invocation
# ---------------------------------------------------------------------------

def test_cli_tran_command(tmp_path):
    """Test running `pycircuit tran` via CLI."""
    net_file = tmp_path / "test_tran.cir"
    net_file.write_text(
        "* Simple RC test\n"
        "V1 1 0 10\n"
        "R1 1 2 1k\n"
        "C1 2 0 1u\n"
        ".end\n"
    )

    exit_code = cli_main(["tran", str(net_file), "--tstop", "0.005", "--dt", "1e-5", "--uic"])
    assert exit_code == 0
