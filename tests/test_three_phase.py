"""
Unit tests for Three-Phase Systems and Blondel's Two-Wattmeter Method.
"""

import math
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import Resistor
from circuit_solver.engine.ac import solve_ac
from circuit_solver.engine.three_phase import (
    build_three_phase_source,
    build_wye_load,
    build_delta_load,
    analyze_two_wattmeters,
)


def test_balanced_wye_wye_system():
    """
    Test balanced 3-phase Y-Y system (Alexander & Sadiku Cap. 12):
    Source: Van = 120 V rms, 60 Hz, positive sequence (A-B-C).
    Load: Zy = 10 + j15 ohms.
    
    Analytical:
    w = 2*pi*60 ~ 376.991 rad/s
    R = 10 ohms, L = 15 / w = 0.0397887 H
    Zy = 10 + j15 -> |Zy| = sqrt(10^2 + 15^2) = sqrt(325) ~ 18.0277 ohms
    Phase current |Ia| = 120 / 18.0277 ~ 6.6564 A
    P_per_phase = |Ia|^2 * R = (6.6564)^2 * 10 ~ 443.08 W
    P_total = 3 * P_per_phase = 1329.23 W
    Q_total = 3 * |Ia|^2 * 15 = 1993.85 VAR
    S_total = sqrt(P^2 + Q^2) = 2396.31 VA
    """
    freq = 60.0
    w = 2.0 * math.pi * freq
    r_val = 10.0
    l_val = 15.0 / w

    sources = build_three_phase_source(
        connection="Y",
        V_rms=120.0,
        sequence="ABC",
        node_a="s_a",
        node_b="s_b",
        node_c="s_c",
        node_neutral="0",
        prefix="Vs",
    )
    load = build_wye_load(
        R=r_val,
        L=l_val,
        node_a="s_a",
        node_b="s_b",
        node_c="s_c",
        node_neutral="load_n",
        prefix="Zy",
    )
    neutral_wire = [Resistor("R_neutral", "load_n", "0", 1e-6)]

    c = Circuit(sources + load + neutral_wire)
    sol = solve_ac(c, frequency=freq)

    v_a = (sol.V("s_a") / np.sqrt(2.0))
    v_b = (sol.V("s_b") / np.sqrt(2.0))
    v_c = (sol.V("s_c") / np.sqrt(2.0))

    # Source amplitudes
    assert np.isclose(abs(v_a), 120.0, atol=1e-3)
    assert np.isclose(abs(v_b), 120.0, atol=1e-3)
    assert np.isclose(abs(v_c), 120.0, atol=1e-3)

    # Line-to-line voltage |Vab| = sqrt(3) * 120 = 207.846 V
    v_ab = v_a - v_b
    assert np.isclose(abs(v_ab), 120.0 * math.sqrt(3.0), atol=1e-3)

    # Phase current
    i_a = (v_a - (sol.V("load_n") / np.sqrt(2.0))) / (r_val + 1j * w * l_val)
    assert np.isclose(abs(i_a), 120.0 / math.sqrt(100.0 + 225.0), atol=1e-3)

    # Neutral voltage should be approximately 0
    assert abs((sol.V("load_n") / np.sqrt(2.0))) < 1e-4


def test_two_wattmeter_method_blondel():
    """
    Test Blondel's Two-Wattmeter Method on a balanced system:
    Source: V_LN = 120 V (V_LL = 207.846 V), 60 Hz.
    Load: Wye load with Z = 20 + j15 ohms.
    |Z| = 25 ohms, theta = atan(15/20) = 36.87 deg (0.6435 rad), PF = cos(theta) = 0.8 lagging.
    
    Current |I_L| = 120 / 25 = 4.8 A.
    Total active power P = 3 * (4.8^2) * 20 = 1382.4 W.
    Total reactive power Q = 3 * (4.8^2) * 15 = 1036.8 VAR.
    """
    freq = 60.0
    w = 2.0 * math.pi * freq
    r_val = 20.0
    l_val = 15.0 / w

    sources = build_three_phase_source(
        connection="Y",
        V_rms=120.0,
        sequence="ABC",
        node_a="s_a",
        node_b="s_b",
        node_c="s_c",
        node_neutral="0",
        prefix="Vs",
    )
    load = build_wye_load(
        R=r_val,
        L=l_val,
        node_a="s_a",
        node_b="s_b",
        node_c="s_c",
        node_neutral="load_n",
        prefix="Zy",
    )
    gnd_res = [Resistor("R_ground", "load_n", "0", 1e-5)]

    c = Circuit(sources + load + gnd_res)
    sol = solve_ac(c, frequency=freq)

    v_a = (sol.V("s_a") / np.sqrt(2.0))
    v_b = (sol.V("s_b") / np.sqrt(2.0))
    v_c = (sol.V("s_c") / np.sqrt(2.0))

    z_phase = r_val + 1j * w * l_val
    i_a = (v_a - (sol.V("load_n") / np.sqrt(2.0))) / z_phase
    i_b = (v_b - (sol.V("load_n") / np.sqrt(2.0))) / z_phase

    v_ac = v_a - v_c
    v_bc = v_b - v_c

    wm_res = analyze_two_wattmeters(
        V_line_ac=v_ac,
        V_line_bc=v_bc,
        I_line_a=i_a,
        I_line_b=i_b,
    )

    expected_p = 3.0 * (4.8 ** 2) * 20.0  # 1382.4 W
    expected_q = 3.0 * (4.8 ** 2) * 15.0  # 1036.8 VAR

    assert np.isclose(wm_res.P_total, expected_p, atol=1e-1)
    assert np.isclose(wm_res.Q_total, expected_q, atol=1e-1)
    assert np.isclose(wm_res.power_factor, 0.8, atol=1e-3)
    assert np.isclose(wm_res.W1 + wm_res.W2, expected_p, atol=1e-1)


def test_balanced_delta_load():
    """
    Test balanced delta load:
    V_LL = 208 V rms, 60 Hz.
    Z_delta = 30 + j40 ohms per phase (|Z_delta| = 50 ohms).
    Phase current in delta branch: |I_phase| = 208 / 50 = 4.16 A.
    Line current magnitude: |I_line| = sqrt(3) * 4.16 = 7.2053 A.
    Total active power P = 3 * |I_phase|^2 * 30 = 3 * (4.16^2) * 30 = 1557.5 W.
    """
    freq = 60.0
    w = 2.0 * math.pi * freq
    r_val = 30.0
    l_val = 40.0 / w
    v_ln = 208.0 / math.sqrt(3.0)

    sources = build_three_phase_source(
        connection="Y",
        V_rms=v_ln,
        sequence="ABC",
        node_a="s_a",
        node_b="s_b",
        node_c="s_c",
        node_neutral="0",
        prefix="Vs",
    )
    load = build_delta_load(
        R=r_val,
        L=l_val,
        node_a="s_a",
        node_b="s_b",
        node_c="s_c",
        prefix="Zd",
    )

    c = Circuit(sources + load)
    sol = solve_ac(c, frequency=freq)

    v_ab = (sol.V("s_a") / np.sqrt(2.0)) - (sol.V("s_b") / np.sqrt(2.0))
    assert np.isclose(abs(v_ab), 208.0, atol=1e-3)

    z_phase = r_val + 1j * w * l_val
    i_ab = v_ab / z_phase
    assert np.isclose(abs(i_ab), 208.0 / 50.0, atol=1e-3)
