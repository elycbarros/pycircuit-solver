"""
test_dc_sweep.py — Unit tests and analytical benchmarks for DC parametric sweep (.dc sweep).

Validates:
- Linear transfer curves (V_out x V_in).
- Resistor load sweep.
- Diode I-V characteristic curve.
- Transistor 2D characteristic curve family (I_C x V_CE across I_B).
- MOSFET transfer curve.
- Netlist .DC directive parsing.
- CLI subcommand 'dc'.
"""

import numpy as np
import pytest

from circuit_solver import (
    Circuit,
    Resistor,
    VoltageSource,
    CurrentSource,
    Diode,
    BJT,
    MOSFET,
    dc_sweep,
)
from circuit_solver.cli import main
from circuit_solver.io.parser import parse_dc_directive


def test_dc_sweep_linear_divider():
    """Transfer curve V_out x V_in for a linear voltage divider."""
    # V1: 0 to 10 V (step 1.0)
    # R1: 1k, R2: 1k -> V(2) = 0.5 * V(1)
    c = Circuit([
        VoltageSource("V1", "1", "0", 0.0),
        Resistor("R1", "1", "2", 1000.0),
        Resistor("R2", "2", "0", 1000.0),
    ])

    res = dc_sweep(c, "V1", 0.0, 10.0, 1.0)
    assert len(res.sweep_values) == 11
    assert pytest.approx(res.sweep_values) == np.linspace(0.0, 10.0, 11)

    v1 = res.V("1")
    v2 = res.V("2")
    assert pytest.approx(v1) == np.linspace(0.0, 10.0, 11)
    assert pytest.approx(v2) == 0.5 * np.linspace(0.0, 10.0, 11)

    # Branch currents: I(R1) = V1 / 2000
    i_r1 = res.I("R1")
    assert pytest.approx(i_r1) == np.linspace(0.0, 10.0, 11) / 2000.0


def test_dc_sweep_resistor_load_curve():
    """Varying a load resistor Rload across a sweep range."""
    # Source: 12V with 50 ohm internal resistance
    c = Circuit([
        VoltageSource("VS", "1", "0", 12.0),
        Resistor("RS", "1", "2", 50.0),
        Resistor("RL", "2", "0", 10.0),
    ])

    res = dc_sweep(c, "RL", 10.0, 100.0, 10.0)
    assert len(res.sweep_values) == 10

    # V_out = 12 * RL / (50 + RL)
    expected_v = 12.0 * res.sweep_values / (50.0 + res.sweep_values)
    assert pytest.approx(res.V("2"), rel=1e-5) == expected_v


def test_dc_sweep_diode_iv_curve():
    """Diode forward I-V characteristic curve."""
    # V1: 0 to 1.0 V (step 0.05 V)
    # R1: 100 ohms
    c = Circuit([
        VoltageSource("V1", "1", "0", 0.0),
        Resistor("R1", "1", "2", 100.0),
        Diode("D1", "2", "0", Is=1e-14, n=1.0),
    ])

    res = dc_sweep(c, "V1", 0.0, 1.5, 0.05)
    id_arr = res.I("D1")
    v2_arr = res.V("2")

    # At low voltage (0.2V), current is virtually zero (< 1 nA)
    idx_low = np.argmin(np.abs(res.sweep_values - 0.2))
    assert id_arr[idx_low] < 1e-9

    # At higher voltage (1.5V), diode is strongly ON with ~0.7V drop
    idx_high = np.argmin(np.abs(res.sweep_values - 1.5))
    assert 0.65 < v2_arr[idx_high] < 0.78
    assert id_arr[idx_high] > 5e-3  # > 5 mA


def test_dc_sweep_bjt_characteristic_curve_family():
    """2D nested sweep: BJT IC x VCE family for multiple base currents IB."""
    c = Circuit([
        VoltageSource("VCE", "c", "0", 0.0),
        CurrentSource("IB", "b", "0", 10e-6),
        BJT("Q1", "c", "b", "0", polarity="NPN", beta=100.0),
    ])

    res = dc_sweep(
        c,
        param_name="VCE",
        start=0.0,
        stop=5.0,
        step=0.2,   # 26 points
        param2_name="IB",
        start2=10e-6,
        stop2=30e-6,
        step2=10e-6, # 3 curves: 10uA, 20uA, 30uA
    )

    assert res.param2_name == "IB"
    assert len(res.sweep2_values) == 3
    assert len(res.sweep_values) == 26

    ic_matrix = res.I("Q1_C")
    assert ic_matrix.shape == (3, 26)

    # At VCE = 5V (active region), IC should equal beta * IB (1mA, 2mA, 3mA)
    assert pytest.approx(ic_matrix[0, -1], rel=0.01) == 1e-3
    assert pytest.approx(ic_matrix[1, -1], rel=0.01) == 2e-3
    assert pytest.approx(ic_matrix[2, -1], rel=0.01) == 3e-3


def test_dc_sweep_mosfet_transfer_curve():
    """NMOS inverter transfer curve: V_out x V_in."""
    # VDD: 5V
    # RD: 2k
    c = Circuit([
        VoltageSource("VDD", "dd", "0", 5.0),
        VoltageSource("VIN", "g", "0", 0.0),
        Resistor("RD", "dd", "d", 2000.0),
        MOSFET("M1", "d", "g", "0", polarity="NMOS", Vth=1.0, Kp=1e-3),
    ])

    res = dc_sweep(c, "VIN", 0.0, 5.0, 0.2)
    vd = res.V("d")

    # In cutoff (VIN < 1.0V), output is at VDD = 5.0V
    cutoff_indices = np.where(res.sweep_values < 0.9)[0]
    for idx in cutoff_indices:
        assert pytest.approx(vd[idx], abs=1e-3) == 5.0

    # With high gate voltage (VIN = 5.0V), output is pulled low
    assert vd[-1] < 1.0


def test_netlist_dc_directive_parser(tmp_path):
    """Test extracting .DC sweep directive from a netlist."""
    netlist = tmp_path / "test_dc.cir"
    netlist.write_text(
        "DC Sweep Test Circuit\n"
        "V1 1 0 5.0\n"
        "R1 1 2 1k\n"
        "R2 2 0 1k\n"
        ".DC V1 0 10 0.5\n"
        ".end\n"
    )

    info = parse_dc_directive(netlist)
    assert info is not None
    assert info["param_name"] == "V1"
    assert info["start"] == 0.0
    assert info["stop"] == 10.0
    assert info["step"] == 0.5


def test_cli_dc_sweep(tmp_path, capsys):
    """Test pycircuit dc command line execution."""
    netlist = tmp_path / "divider.cir"
    netlist.write_text(
        "Divider\n"
        "V1 1 0 10.0\n"
        "R1 1 2 1000\n"
        "R2 2 0 1000\n"
        ".end\n"
    )

    ret = main(["dc", str(netlist), "--sweep", "V1", "0", "10", "2", "--node", "2"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "DC Parametric Sweep Summary" in out
    assert "Primary parameter   : V1" in out
