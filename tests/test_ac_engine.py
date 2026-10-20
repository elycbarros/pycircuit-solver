"""
test_ac_engine.py — Testes analíticos para o motor de regime permanente CA (Alexander & Sadiku, Cap. 9 & 10).
"""

import numpy as np
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Capacitor,
    Inductor,
    Resistor,
    VoltageSource,
)
from circuit_solver.engine.ac import ac_sweep, solve_ac
from circuit_solver.visualization.plots import plot_bode


def test_rc_lowpass_cutoff_frequency():
    """Filtro passa-baixas RC:
    R = 1kΩ, C = 1μF.
    Frequência de corte fc = 1 / (2*pi*R*C) ≈ 159.155 Hz.
    Vin = 10 V ∠ 0°.
    Em fc:
    - Vout_mag = 10 / sqrt(2) ≈ 7.0711 V
    - Vout_phase ≈ -45.0°
    """
    R_val = 1000.0
    C_val = 1e-6
    fc = 1.0 / (2.0 * np.pi * R_val * C_val)

    elements = [
        VoltageSource("Vin", "in", "0", 0.0, ac_mag=10.0, ac_phase=0.0),
        Resistor("R1", "in", "out", R_val),
        Capacitor("C1", "out", "0", C_val),
    ]
    circuit = Circuit(elements)
    sol = solve_ac(circuit, frequency=fc)

    assert sol.V_mag("in") == pytest.approx(10.0)
    assert sol.V_phase_deg("in") == pytest.approx(0.0)

    # Na frequência de corte: |Vout| = 10 / sqrt(2)
    assert sol.V_mag("out") == pytest.approx(10.0 / np.sqrt(2.0), rel=1e-3)
    assert sol.V_rms("out") == pytest.approx((10.0 / np.sqrt(2.0)) / np.sqrt(2.0), rel=1e-3)

    # Fase deve ser -45 graus
    assert sol.V_phase_deg("out") == pytest.approx(-45.0, abs=0.1)

    # Potência complexa no capacitor deve ser puramente reativa (P ~ 0, Q < 0)
    s_cap = sol.complex_power("C1")
    assert s_cap.real == pytest.approx(0.0, abs=1e-6)
    assert s_cap.imag < 0.0  # Capacitor absorve Q negativo (fornece reativo)


def test_rlc_series_resonance():
    """Circuito RLC série sob ressonância:
    Vs = 120V ∠ 0°
    R = 20Ω, L = 50mH, C = 20μF
    omega_0 = 1 / sqrt(L*C) = 1 / sqrt(0.05 * 20e-6) = 1000 rad/s -> f0 = 1000 / (2*pi) ≈ 159.155 Hz.
    Na ressonância: Z_eq = R = 20Ω.
    Corrente I = 120 / 20 = 6A ∠ 0°.
    Tensão no resistor Vr = 120V ∠ 0°.
    Tensão no indutor Vl = j*omega*L * I = j*50 * 6 = 300V ∠ 90°.
    Tensão no capacitor Vc = -j / (omega*C) * I = -j*50 * 6 = 300V ∠ -90°.
    """
    R_val = 20.0
    L_val = 50e-3
    C_val = 20e-6
    f0 = 1000.0 / (2.0 * np.pi)

    elements = [
        VoltageSource("Vs", "1", "0", 0.0, ac_mag=120.0, ac_phase=0.0),
        Resistor("R1", "1", "2", R_val),
        Inductor("L1", "2", "3", L_val),
        Capacitor("C1", "3", "0", C_val),
    ]
    circuit = Circuit(elements)
    sol = solve_ac(circuit, frequency=f0)

    # Corrente através do indutor
    i_ind = sol.I_inductor("L1")
    assert abs(i_ind) == pytest.approx(6.0, rel=1e-3)
    assert np.degrees(np.angle(i_ind)) == pytest.approx(0.0, abs=0.1)

    # Tensão no nó 3 (tensão sobre o capacitor): 300V ∠ -90°
    assert sol.V_mag("3") == pytest.approx(300.0, rel=1e-3)
    assert sol.V_phase_deg("3") == pytest.approx(-90.0, abs=0.1)

    # Resumo formatado
    summary_txt = sol.summary()
    assert "AC Phasor Solution" in summary_txt
    assert "Node Voltages (Phasors)" in summary_txt


def test_ac_frequency_sweep_and_bode():
    """Teste de varredura em frequência e geração de diagrama de Bode."""
    elements = [
        VoltageSource("Vin", "in", "0", 0.0, ac_mag=1.0, ac_phase=0.0),
        Resistor("R1", "in", "out", 1000.0),
        Capacitor("C1", "out", "0", 1e-6),
    ]
    circuit = Circuit(elements)
    freqs = np.logspace(1, 4, 30)

    frequencies, mag_db, phase_deg = ac_sweep(circuit, freqs, target_node="out")
    assert len(mag_db) == 30
    assert len(phase_deg) == 30

    # Baixas frequências (10 Hz): ganho ~ 0 dB (1.0), fase ~ 0°
    assert mag_db[0] == pytest.approx(0.0, abs=0.5)
    assert phase_deg[0] == pytest.approx(0.0, abs=5.0)

    # Altas frequências (10 kHz): ganho << 0 dB, fase ~ -90°
    assert mag_db[-1] < -20.0
    assert phase_deg[-1] == pytest.approx(-90.0, abs=5.0)

    # Test plot_bode
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_bode(frequencies, mag_db, phase_deg, show=False)
    assert fig is not None
