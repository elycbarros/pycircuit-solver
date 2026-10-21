"""
test_small_signal_ac.py — Small-signal AC phasor analysis with semiconductors.

Validates Q-point linearization and AC frequency response for:
1. Dynamic resistance of forward-biased PN junction diodes.
2. Common-Emitter BJT amplifier with coupling/bypass capacitors and Bode sweep.
3. Common-Source MOSFET amplifier small-signal voltage gain.
4. Pedagogical Q-point and hybrid-pi parameters inspection report.
5. CLI pycircuit op command.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    Capacitor,
    VoltageSource,
    Diode,
    BJT,
    MOSFET,
)
from circuit_solver.engine.ac import solve_ac, ac_sweep
from circuit_solver.engine.small_signal import (
    compute_operating_point,
    build_small_signal_circuit,
)
from circuit_solver.cli import main


def test_diode_small_signal_dynamic_resistance():
    """Verify diode AC dynamic resistance r_d = n * Vt / I_D and voltage divider.

    DC: Vdc = 5.0 V in series with R = 1.0 kΩ and Diode.
    I_D ≈ (5.0 - 0.7) / 1k ≈ 4.3 mA.
    r_d = Vt / I_D ≈ 25.85 mV / 4.3 mA ≈ 6.0 Ω.
    AC: Vac = 100 mV peak at 1 kHz.
    v_diode_ac = Vac * (r_d / (R + r_d)) ≈ 100 mV * (6.0 / 1006.0) ≈ 0.596 mV.
    """
    netlist = """
    Diode Small Signal Voltage Divider
    Vin in 0 DC 5.0 AC 0.1
    R1 in out 1000.0
    D1 out 0
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    op = compute_operating_point(circuit)
    d_qp = op.diodes["D1"]

    assert d_qp.state == "FORWARD BIASED"
    assert 0.6 < d_qp.v_d < 0.8
    assert 3.5e-3 < d_qp.i_d < 5.0e-3
    assert 4.0 < d_qp.r_d < 10.0

    # AC solution at 1 kHz
    sol_ac = solve_ac(circuit, 1000.0)
    v_out_ac = sol_ac.V_mag("out")

    expected_ac = 0.1 * (d_qp.r_d / (1000.0 + d_qp.r_d))
    assert pytest.approx(v_out_ac, rel=0.05) == expected_ac


def test_bjt_common_emitter_amplifier_gain_and_bode():
    """Verify BJT Common-Emitter amplifier midband gain and AC frequency sweep.

    Classic Sedra & Smith / Alexander & Sadiku CE Amplifier:
    Vcc = 12 V
    R1 = 47 kΩ, R2 = 10 kΩ (V_B ≈ 12 * 10 / 57 ≈ 2.1 V -> V_E ≈ 1.4 V)
    Re = 1 kΩ (I_E ≈ I_C ≈ 1.4 mA)
    Rc = 3.3 kΩ (V_C ≈ 12 - 1.4m * 3.3k ≈ 7.4 V -> V_CE ≈ 6.0 V: Forward Active)
    gm ≈ Ic / Vt ≈ 1.4m / 25.85m ≈ 54.1 mA/V
    Midband Gain: Av = v_out / v_in ≈ -gm * Rc ≈ -54.1m * 3.3k ≈ -178 (≈ 45 dB)
    Coupling capacitor Cin = 10 μF provides high-pass cutoff.
    """
    netlist = """
    BJT Common Emitter Amplifier
    Vcc vcc 0 DC 12.0
    Vin in 0 DC 0.0 AC 1.0
    Cin in b 10u
    R1 vcc b 47k
    R2 b 0 10k
    Rc vcc c 3.3k
    Re e 0 1k
    Ce e 0 100u
    Q1 c b e NPN BETA=100
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    op = compute_operating_point(circuit)
    b_qp = op.bjts["Q1"]

    assert b_qp.region == "FORWARD ACTIVE"
    assert 0.030 < b_qp.g_m < 0.070  # ~50 mA/V
    assert 1000.0 < b_qp.r_pi < 4000.0

    # Midband AC solution at 10 kHz (capacitors are low impedance)
    sol_ac = solve_ac(circuit, 10000.0)
    v_in_ac = sol_ac.V_mag("in")
    v_out_ac = sol_ac.V_mag("c")
    gain = v_out_ac / v_in_ac

    # Small-signal gain magnitude should be in the range 100 to 200
    assert 100.0 < gain < 200.0

    # Bode sweep from 1 Hz to 100 kHz
    freqs = np.logspace(0, 5, 20)
    frequencies, mag_db, phase_deg = ac_sweep(circuit, freqs, target_node="c")

    # At 1 Hz, Cin blocks the signal (gain is very low)
    assert mag_db[0] < 20.0
    # At 10 kHz (midband), gain is around 40-45 dB
    assert mag_db[-1] > 35.0
    assert mag_db[-1] > mag_db[0] + 15.0


def test_mosfet_common_source_amplifier():
    """Verify MOSFET Common-Source amplifier small-signal voltage gain.

    Vdd = 10 V
    Rd = 4.0 kΩ
    Vgs bias = 2.5 V
    NMOS: Vth = 1.0 V, Kp = 2e-3 A/V^2
    Vov = 2.5 - 1.0 = 1.5 V
    Id = 0.5 * 2m * (1.5)^2 = 2.25 mA
    gm = Kp * Vov = 2m * 1.5 = 3.0 mA/V
    Vd = 10 - 2.25m * 4k = 1.0 V (Saturation since Vds = 1.0V >= Vov = 1.5V? Wait:
    To ensure saturation: Vdd=12V, Rd=2k -> Vd = 12 - 4.5 = 7.5V >= 1.5V).
    """
    netlist = """
    MOSFET Common Source Stage
    Vdd vdd 0 DC 12.0
    Vbias g 0 DC 2.5 AC 10m
    Rd vdd d 2000.0
    M1 d g 0 NMOS VTH=1.0 KP=2e-3
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    op = compute_operating_point(circuit)
    m_qp = op.mosfets["M1"]

    assert m_qp.region == "SATURATION"
    assert pytest.approx(m_qp.g_m, rel=0.05) == 3.0e-3  # 3 mA/V

    # AC solution at 1 kHz: Av = -gm * Rd = -3mA/V * 2kΩ = -6.0
    sol_ac = solve_ac(circuit, 1000.0)
    v_in = sol_ac.V_mag("g")
    v_out = sol_ac.V_mag("d")
    gain = v_out / v_in

    assert pytest.approx(gain, rel=0.05) == 6.0


def test_pedagogical_operating_point_report():
    """Verify operating point summary formatting and theoretical details."""
    netlist = """
    Transistor Bias Test
    Vcc 1 0 DC 10.0
    R1 1 2 10k
    R2 2 0 10k
    Rc 1 3 1k
    Q1 3 2 0 NPN BETA=100
    .END
    """
    circuit = Circuit.from_netlist(netlist)
    op = compute_operating_point(circuit)
    summary = op.summary()

    assert "RELATÓRIO DO PONTO DE OPERAÇÃO DC (Q-POINT)" in summary
    assert "TRANSISTORES BIPOLARES (BJT" in summary
    assert "Q1 [NPN]" in summary
    assert "g_m =" in summary
    assert "r_π =" in summary


def test_cli_op_command(tmp_path: Path, capsys):
    """Verify CLI 'pycircuit op' prints the pedagogical operating point report."""
    cir_file = tmp_path / "cli_op_test.cir"
    cir_file.write_text(
        """
        CLI OP Test
        V1 in 0 DC 5.0
        R1 in out 1k
        D1 out 0
        .END
        """,
        encoding="utf-8",
    )

    ret = main(["op", str(cir_file)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "RELATÓRIO DO PONTO DE OPERAÇÃO DC (Q-POINT)" in captured.out
    assert "DIODOS E ZENERS" in captured.out
    assert "D1" in captured.out
    assert "FORWARD BIASED" in captured.out
