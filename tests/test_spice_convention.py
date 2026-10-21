"""Convenção SPICE para fontes de corrente (I, G) e para formas de onda (PULSE, SIN).

Os valores esperados vêm da definição do SPICE (ngspice manual, cap. de fontes):

* I n+ n- valor: a corrente sai de n+, passa pela fonte e entra em n-.
  Logo ``I1 0 1 1m`` injeta 1 mA no nó 1.
* G n+ n- nc+ nc- gm: i = gm * (V(nc+) - V(nc-)), com o mesmo sentido de n+ para n-.
  Logo ``G1 0 2 1 0 1m`` injeta corrente no nó 2.
* F n+ n- Vnome ganho: sentido de n+ para n-, mantido como já estava.

Um caso de referência por fonte, resolvido analiticamente. A conferência com o
ngspice fica para o Mac (``ngspice -b arquivo.cir``).
"""

import pytest

from circuit_solver import Circuit, solve


def _v(netlist: str, node: str) -> float:
    return solve(Circuit.from_netlist(netlist)).voltages[node]


# ---------------------------------------------------------------------------
# Fonte de corrente independente I
# ---------------------------------------------------------------------------

def test_I_injects_into_negative_terminal_as_spice():
    # 1 mA sai de 0 e entra no nó 1 -> V(1) = +1 V sobre 1 kΩ
    assert _v("t\nI1 0 1 1m\nR1 1 0 1k\n.end\n", "1") == pytest.approx(1.0)


def test_I_reversed_terminals_gives_negative_voltage():
    assert _v("t\nI1 1 0 1m\nR1 1 0 1k\n.end\n", "1") == pytest.approx(-1.0)


def test_I_with_DC_keyword_same_convention():
    assert _v("t\nI1 0 1 DC 2m\nR1 1 0 1k\n.end\n", "1") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Fonte controlada por tensão G (VCCS)
# ---------------------------------------------------------------------------

def test_G_injects_into_negative_terminal_as_spice():
    # V(1) = 1 V, gm = 1 mS. i = 1 mA, sai de 0 e entra no nó 2 -> V(2) = +1 V
    netlist = "t\nV1 1 0 1\nR1 1 0 1k\nG1 0 2 1 0 1m\nR2 2 0 1k\n.end\n"
    assert _v(netlist, "2") == pytest.approx(1.0)


def test_G_reversed_terminals_gives_negative_voltage():
    netlist = "t\nV1 1 0 1\nR1 1 0 1k\nG1 2 0 1 0 1m\nR2 2 0 1k\n.end\n"
    assert _v(netlist, "2") == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Fonte controlada por corrente F (CCCS): já segue SPICE, precisa continuar seguindo
# ---------------------------------------------------------------------------

def test_F_keeps_spice_convention():
    # I(V0) = +1 mA (do nó 4 para 0, pelo V0). F injeta 2*I(V0) no nó 3 -> V(3) = +2 V
    netlist = "t\nV1 1 0 1\nR1 1 4 1k\nV0 4 0 0\nF1 0 3 V0 2\nR3 3 0 1k\n.end\n"
    assert _v(netlist, "3") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Ponto de operação (DC) com PULSE e SIN
# ---------------------------------------------------------------------------

def test_pulse_without_dc_uses_v1_as_operating_point():
    # Ponto de operação de PULSE(0 1 ...) é v1 = 0 V, não o último token numérico
    netlist = "t\nV1 1 0 PULSE(0 1 0 1n 1n 10m 20m)\nR1 1 0 1k\n.end\n"
    assert _v(netlist, "1") == pytest.approx(0.0, abs=1e-12)


def test_pulse_with_v1_nonzero_uses_v1():
    netlist = "t\nV1 1 0 PULSE(2 5 0 1n 1n 10m 20m)\nR1 1 0 1k\n.end\n"
    assert _v(netlist, "1") == pytest.approx(2.0)


def test_explicit_DC_overrides_pulse_for_operating_point():
    netlist = "t\nV1 1 0 DC 5 PULSE(0 1 0 1n 1n 10m 20m)\nR1 1 0 1k\n.end\n"
    assert _v(netlist, "1") == pytest.approx(5.0)


def test_explicit_DC_zero_with_pulse_is_zero():
    # Caso que o parser atual lia como 0.01 V (último token numérico do PULSE)
    netlist = "t\nV1 1 0 DC 0 PULSE(0 1 0 1n 1n 10m 20m)\nR1 1 0 1k\n.end\n"
    assert _v(netlist, "1") == pytest.approx(0.0, abs=1e-12)


def test_sin_operating_point_is_offset():
    netlist = "t\nV1 1 0 SIN(2 1 1k)\nR1 1 0 1k\n.end\n"
    assert _v(netlist, "1") == pytest.approx(2.0)


def test_current_pulse_without_dc_gives_zero_operating_point():
    netlist = "t\nI1 0 1 PULSE(0 2m 0 1n 1n 1m 2m)\nR1 1 0 1k\n.end\n"
    assert _v(netlist, "1") == pytest.approx(0.0, abs=1e-12)
