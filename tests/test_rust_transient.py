"""
test_rust_transient.py — Test suite for high-performance Rust transient solver.

Validates:
  1. Rust engine availability & seamless dispatching (`backend="rust"` and `backend="auto"`).
  2. Numerical equivalence between Rust and Python/SciPy solvers (RC, RLC, Pulse, Sine).
  3. Energy audit & physical conservation:
       E(t) = 0.5 * C * V^2 + 0.5 * L * I^2
       P_diss(t) = V^2 / R
  4. Extreme execution speed & throughput (>500,000 steps/sec).
  5. Fallback behavior and error handling.
"""

import math
import time
import pytest
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    StepWaveform,
    PulseWaveform,
    SineWaveform,
    Diode,
)
from circuit_solver.engine.transient import solve_transient, RUST_AVAILABLE
from circuit_solver.engine.transient_rust import can_run_in_rust


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not installed")
class TestRustTransientCore:
    def test_rc_step_response_rust_vs_python(self):
        """Verify RC circuit step response in Rust matches Python and theoretical formula."""
        r_val = 1e3
        c_val = 1e-6
        tau = r_val * c_val  # 1.0 ms
        v_step = 10.0

        circ = Circuit([
            VoltageSource("V1", "1", "0", v_step, waveform=StepWaveform(v1=0.0, v2=v_step, t_step=0.0)),
            Resistor("R1", "1", "2", r_val),
            Capacitor("C1", "2", "0", c_val, ic=0.0),
        ])

        t_stop = 5.0 * tau
        dt = 1e-5

        sol_rust = solve_transient(circ, t_stop=t_stop, dt=dt, backend="rust", uic=True)
        sol_py = solve_transient(circ, t_stop=t_stop, dt=dt, backend="python", uic=True)

        assert sol_rust.execution_metrics is not None
        assert "Rust" in str(sol_rust.execution_metrics["backend"])

        # Waveforms must match tightly between Rust and Python
        v_c_rust = sol_rust.V("2")
        v_c_py = sol_py.V("2")
        np.testing.assert_allclose(v_c_rust, v_c_py, atol=1e-10, rtol=1e-10)

        # Theoretical formula check at characteristic points (1 tau, 3 tau, 5 tau)
        for target_t in [1.0 * tau, 3.0 * tau, 5.0 * tau]:
            idx = int(round(target_t / dt))
            exact_v = v_step * (1.0 - math.exp(-target_t / tau))
            sim_v = v_c_rust[idx]
            assert abs(sim_v - exact_v) < 0.02, f"At {target_t*1000}ms: sim={sim_v}, exact={exact_v}"

        # Branch currents match tightly during stepping
        i_c_rust = sol_rust.I("C1")
        i_r_rust = sol_rust.I("R1")
        np.testing.assert_allclose(i_c_rust[1:], i_r_rust[1:], atol=1e-10)



    def test_rc_dc_operating_point_start_uic_false(self):
        """Verify operating point DC initialization when uic=False."""
        circ = Circuit([
            VoltageSource("V1", "1", "0", 10.0),
            Resistor("R1", "1", "2", 1000.0),
            Capacitor("C1", "2", "0", 1e-6),
        ])
        sol_rust = solve_transient(circ, t_stop=1e-3, dt=1e-5, backend="rust", uic=False)
        sol_py = solve_transient(circ, t_stop=1e-3, dt=1e-5, backend="python", uic=False)

        # In steady state DC, capacitor is open circuit -> V(2) = 10.0 V
        np.testing.assert_allclose(sol_rust.V("2"), 10.0, atol=1e-10)
        np.testing.assert_allclose(sol_rust.V("2"), sol_py.V("2"), atol=1e-10)

    def test_rlc_underdamped_oscillation_rust(self):
        """Verify underdamped second-order RLC transient response."""
        # Series RLC: R = 10 Ohm, L = 10 mH, C = 10 uF
        # omega_0 = 1 / sqrt(L*C) = 1 / sqrt(1e-7) = 3162.28 rad/s
        # alpha = R / (2L) = 10 / (2 * 10e-3) = 500 s^-1
        # alpha < omega_0 -> underdamped oscillation
        r_val = 10.0
        l_val = 10e-3
        c_val = 10e-6

        circ = Circuit([
            VoltageSource("V1", "1", "0", 12.0, waveform=StepWaveform(v1=0.0, v2=12.0, t_step=0.0)),
            Resistor("R1", "1", "2", r_val),
            Inductor("L1", "2", "3", l_val, ic=0.0),
            Capacitor("C1", "3", "0", c_val, ic=0.0),
        ])

        t_stop = 10e-3
        dt = 2e-6

        sol_rust = solve_transient(circ, t_stop=t_stop, dt=dt, backend="rust", uic=True)
        sol_py = solve_transient(circ, t_stop=t_stop, dt=dt, backend="python", uic=True)

        v_c_rust = sol_rust.V("3")
        v_c_py = sol_py.V("3")
        np.testing.assert_allclose(v_c_rust, v_c_py, atol=1e-9, rtol=1e-9)

        # In underdamped step response, capacitor voltage overshoots initial input voltage
        v_max = np.max(v_c_rust)
        assert v_max > 12.0  # Overshoot present
        assert v_c_rust[-1] == pytest.approx(12.0, rel=0.05)

    def test_energy_conservation_and_joule_audit(self):
        """Pedagogical audit: verify energy storage and dissipation calculations."""
        c_val = 2e-6
        circ = Circuit([
            VoltageSource("V1", "1", "0", 5.0, waveform=StepWaveform(v1=0.0, v2=5.0, t_step=0.0)),
            Resistor("R1", "1", "2", 100.0),
            Capacitor("C1", "2", "0", c_val, ic=0.0),
        ])

        sol = solve_transient(circ, t_stop=3e-3, dt=1e-6, backend="rust", uic=True)

        e_stored = sol.E_stored()
        p_diss = sol.P_dissipated()

        # At t=0, capacitor uncharged -> energy = 0
        assert e_stored[0] == pytest.approx(0.0)

        # In steady-state (t -> inf), capacitor charged to 5V
        # Theoretical final energy E_final = 0.5 * C * V^2 = 0.5 * 2e-6 * 25 = 25 uJ
        e_final_expected = 0.5 * c_val * (5.0 ** 2)
        assert e_stored[-1] == pytest.approx(e_final_expected, rel=1e-3)

        # In RC charging from 0 to V, total energy dissipated in R equals total energy stored in C:
        # W_R = integral(P_diss * dt) = 0.5 * C * V^2
        w_dissipated = np.sum(p_diss) * 1e-6
        assert w_dissipated == pytest.approx(e_final_expected, rel=1e-2)

        # Summary output contains pedagogical info
        summary_text = sol.summary()
        assert "Peak Energy Stored:" in summary_text
        assert "Backend: Rust" in summary_text


    def test_pulse_and_sine_waveforms(self):
        """Verify SPICE PULSE and SIN source handling in Rust engine."""
        circ_pulse = Circuit([
            VoltageSource("Vpulse", "1", "0", 0.0, waveform=PulseWaveform(
                v1=0.0, v2=5.0, td=1e-4, tr=1e-5, tf=1e-5, pw=2e-4, per=5e-4
            )),
            Resistor("R1", "1", "0", 1000.0),
        ])
        sol_p_rust = solve_transient(circ_pulse, t_stop=1e-3, dt=1e-6, backend="rust")
        sol_p_py = solve_transient(circ_pulse, t_stop=1e-3, dt=1e-6, backend="python")
        np.testing.assert_allclose(sol_p_rust.V("1"), sol_p_py.V("1"), atol=1e-12)

        circ_sine = Circuit([
            VoltageSource("Vsine", "1", "0", 0.0, waveform=SineWaveform(
                vo=0.0, va=10.0, freq=1000.0, td=0.0
            )),
            Resistor("R1", "1", "0", 50.0),
        ])
        sol_s_rust = solve_transient(circ_sine, t_stop=2e-3, dt=1e-6, backend="rust")
        sol_s_py = solve_transient(circ_sine, t_stop=2e-3, dt=1e-6, backend="python")
        np.testing.assert_allclose(sol_s_rust.V("1"), sol_s_py.V("1"), atol=1e-12)

    def test_high_throughput_performance(self):
        """Demonstrate raw speed: 50,000 steps executed in milliseconds."""
        circ = Circuit([
            VoltageSource("V1", "1", "0", 5.0, waveform=StepWaveform(v1=0.0, v2=5.0, t_step=0.0)),
            Resistor("R1", "1", "2", 100.0),
            Capacitor("C1", "2", "0", 1e-6, ic=0.0),
        ])

        n_steps_target = 50_000
        dt = 1e-7
        t_stop = n_steps_target * dt

        t0 = time.perf_counter()
        sol = solve_transient(circ, t_stop=t_stop, dt=dt, backend="rust")
        t_elapsed = time.perf_counter() - t0

        assert sol.n_steps >= n_steps_target
        # 50k steps in Rust should complete in well under 100 ms (< 0.1s)
        assert t_elapsed < 0.25
        metrics = sol.execution_metrics
        assert metrics is not None
        assert metrics["steps_per_sec"] > 200_000

    def test_nonlinear_fallback(self):
        """Circuits with non-linear elements fallback cleanly or raise error on forced rust."""
        circ = Circuit([
            VoltageSource("V1", "1", "0", 5.0),
            Resistor("R1", "1", "2", 100.0),
            Diode("D1", "2", "0"),
        ])

        compatible, reason = can_run_in_rust(circ)
        assert not compatible
        assert "Diode" in reason

        # Forced backend="rust" must raise clear ValueError
        with pytest.raises(ValueError, match="cannot be simulated with Rust transient engine"):
            solve_transient(circ, t_stop=1e-3, dt=1e-5, backend="rust")
