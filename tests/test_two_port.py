"""
Unit tests for Two-Port Network Analysis (Z, Y, H, ABCD parameters).
"""

import math
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import Resistor, Capacitor, Inductor
from circuit_solver.engine.two_port import analyze_two_port


def test_t_network_resistive_z_parameters():
    """
    Test classical T-network (Alexander & Sadiku / Hayt):
    R1 = 20 ohms (port 1 series arm, n1 -> n_mid)
    R2 = 40 ohms (port 2 series arm, n2 -> n_mid)
    R3 = 30 ohms (shunt arm, n_mid -> 0)
    
    Analytical Z-parameters:
    Z11 = V1/I1 | I2=0 = R1 + R3 = 50 ohms
    Z12 = V1/I2 | I1=0 = R3 = 30 ohms
    Z21 = V2/I1 | I2=0 = R3 = 30 ohms
    Z22 = V2/I2 | I1=0 = R2 + R3 = 70 ohms
    """
    elements = [
        Resistor("R1", "n1", "n_mid", 20.0),
        Resistor("R2", "n2", "n_mid", 40.0),
        Resistor("R3", "n_mid", "0", 30.0),
    ]
    c = Circuit(elements, allowed_open_nodes=["n1", "n2"])

    res = analyze_two_port(c, port1=("n1", "0"), port2=("n2", "0"))

    assert np.isclose(res.Z[0, 0].real, 50.0, atol=1e-5)
    assert np.isclose(res.Z[0, 1].real, 30.0, atol=1e-5)
    assert np.isclose(res.Z[1, 0].real, 30.0, atol=1e-5)
    assert np.isclose(res.Z[1, 1].real, 70.0, atol=1e-5)

    assert res.is_reciprocal
    assert not res.is_symmetric  # Z11 != Z22


def test_t_network_abcd_and_y_parameters():
    """
    For the above T-network (Z11=50, Z12=30, Z21=30, Z22=70):
    det(Z) = 50*70 - 30*30 = 3500 - 900 = 2600.
    
    ABCD:
    A = Z11/Z21 = 50/30 = 5/3 ~ 1.66667
    B = det(Z)/Z21 = 2600/30 = 260/3 ~ 86.66667
    C = 1/Z21 = 1/30 ~ 0.03333
    D = Z22/Z21 = 70/30 = 7/3 ~ 2.33333
    AD - BC = (5/3)*(7/3) - (260/3)*(1/30) = 35/9 - 26/9 = 9/9 = 1.0 (reciprocity check)
    """
    elements = [
        Resistor("R1", "n1", "n_mid", 20.0),
        Resistor("R2", "n2", "n_mid", 40.0),
        Resistor("R3", "n_mid", "0", 30.0),
    ]
    c = Circuit(elements, allowed_open_nodes=["n1", "n2"])

    res = analyze_two_port(c, port1=("n1", "0"), port2=("n2", "0"))

    A, B = res.ABCD[0, 0], res.ABCD[0, 1]
    C, D = res.ABCD[1, 0], res.ABCD[1, 1]

    assert np.isclose(A.real, 5.0 / 3.0, atol=1e-5)
    assert np.isclose(B.real, 260.0 / 3.0, atol=1e-5)
    assert np.isclose(C.real, 1.0 / 30.0, atol=1e-5)
    assert np.isclose(D.real, 7.0 / 3.0, atol=1e-5)
    assert np.isclose(A * D - B * C, 1.0, atol=1e-5)

    # Y parameters: inv(Z)
    assert np.isclose(res.Y[0, 0].real, 7.0 / 260.0, atol=1e-5)
    assert np.isclose(res.Y[0, 1].real, -3.0 / 260.0, atol=1e-5)
    assert np.isclose(res.Y[1, 0].real, -3.0 / 260.0, atol=1e-5)
    assert np.isclose(res.Y[1, 1].real, 5.0 / 260.0, atol=1e-5)


def test_pi_network_resistive():
    """
    Pi network:
    Ra = 12 ohms (port 1 to gnd)
    Rb = 6 ohms (port 1 to port 2)
    Rc = 4 ohms (port 2 to gnd)
    
    Y-parameters:
    Y11 = 1/Ra + 1/Rb = 1/12 + 1/6 = 3/12 = 0.25 S
    Y12 = -1/Rb = -1/6 S ~ -0.16667 S
    Y21 = -1/Rb = -1/6 S ~ -0.16667 S
    Y22 = 1/Rc + 1/Rb = 1/4 + 1/6 = 5/12 S ~ 0.41667 S
    """
    elements = [
        Resistor("Ra", "n1", "0", 12.0),
        Resistor("Rb", "n1", "n2", 6.0),
        Resistor("Rc", "n2", "0", 4.0),
    ]
    c = Circuit(elements, allowed_open_nodes=["n1", "n2"])

    res = analyze_two_port(c, port1=("n1", "0"), port2=("n2", "0"))

    assert np.isclose(res.Y[0, 0].real, 0.25, atol=1e-5)
    assert np.isclose(res.Y[0, 1].real, -1.0 / 6.0, atol=1e-5)
    assert np.isclose(res.Y[1, 0].real, -1.0 / 6.0, atol=1e-5)
    assert np.isclose(res.Y[1, 1].real, 5.0 / 12.0, atol=1e-5)


def test_two_port_input_and_output_impedance():
    """
    Test Zin with load and Zout with source.
    For symmetric T with R1=R2=20, R3=40:
    Z11 = 60, Z12=40, Z21=40, Z22=60.
    ABCD:
    A = 60/40 = 1.5, B = (3600-1600)/40 = 50, C = 1/40 = 0.025, D = 1.5
    If ZL = 100 ohms:
    Zin = (A*ZL + B) / (C*ZL + D) = (1.5*100 + 50) / (0.025*100 + 1.5) = (150+50)/(2.5+1.5) = 200/4 = 50 ohms.
    Direct check: R1 + (R3 || (R2 + ZL)) = 20 + (40 || 120) = 20 + 30 = 50 ohms!
    """
    elements = [
        Resistor("R1", "n1", "mid", 20.0),
        Resistor("R2", "n2", "mid", 20.0),
        Resistor("R3", "mid", "0", 40.0),
    ]
    c = Circuit(elements, allowed_open_nodes=["n1", "n2"])

    res = analyze_two_port(c, port1=("n1", "0"), port2=("n2", "0"))
    zin = res.input_impedance(100.0)
    assert np.isclose(zin.real, 50.0, atol=1e-5)

    # Check Zout with ZS = 60 ohms:
    # Direct check: R2 + (R3 || (R1 + ZS)) = 20 + (40 || 80) = 20 + 80/3 = 140/3 ~ 46.6667 ohms
    zout = res.output_impedance(60.0)
    assert np.isclose(zout.real, 140.0 / 3.0, atol=1e-5)


def test_two_port_ac_reactive():
    """
    AC Two-port network with capacitor and inductor at f = 1000 Hz:
    w = 2*pi*1000 rad/s
    Series L = 10 mH (XL = j*w*L = j*62.83185)
    Shunt C = 10 uF (XC = 1/(j*w*C) = -j / (2*pi*1000 * 10e-6) = -j*15.91549)
    """
    freq = 1000.0
    w = 2.0 * math.pi * freq
    L_val = 10e-3
    C_val = 10e-6
    Z_L = 1j * w * L_val
    Z_C = 1.0 / (1j * w * C_val)

    elements = [
        Inductor("L1", "n1", "n2", L_val),
        Capacitor("C1", "n2", "0", C_val),
    ]
    c = Circuit(elements, allowed_open_nodes=["n1", "n2"])

    res = analyze_two_port(c, port1=("n1", "0"), port2=("n2", "0"), frequency=freq)

    # Z11 = Z_L + Z_C
    # Z12 = Z_C
    # Z21 = Z_C
    # Z22 = Z_C
    assert np.isclose(res.Z[0, 0], Z_L + Z_C, atol=1e-4)
    assert np.isclose(res.Z[0, 1], Z_C, atol=1e-4)
    assert np.isclose(res.Z[1, 0], Z_C, atol=1e-4)
    assert np.isclose(res.Z[1, 1], Z_C, atol=1e-4)


def test_two_port_cascade():
    """
    Test cascading two networks:
    Network 1: Series resistor R = 10 -> ABCD = [[1, 10], [0, 1]]
    Network 2: Shunt resistor R = 20 -> ABCD = [[1, 0], [1/20, 1]]
    Cascade = N1 x N2 = [[1 + 10/20, 10], [1/20, 1]] = [[1.5, 10], [0.05, 1]]
    """
    c1 = Circuit([Resistor("R1", "n1", "n2", 10.0), Resistor("Rgnd1", "n2", "0", 1e9)], allowed_open_nodes=["n1", "n2"])
    res1 = analyze_two_port(c1, ("n1", "0"), ("n2", "0"))

    c2 = Circuit([Resistor("Rwire", "n1", "n2", 1e-9), Resistor("Rshunt", "n2", "0", 20.0)], allowed_open_nodes=["n1", "n2"])
    res2 = analyze_two_port(c2, ("n1", "0"), ("n2", "0"))

    casc = res1.cascade(res2)
    assert np.isclose(casc.ABCD[0, 0].real, 1.5, atol=1e-3)
    assert np.isclose(casc.ABCD[0, 1].real, 10.0, atol=1e-3)
    assert np.isclose(casc.ABCD[1, 0].real, 0.05, atol=1e-3)
    assert np.isclose(casc.ABCD[1, 1].real, 1.0, atol=1e-3)
