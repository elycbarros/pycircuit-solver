"""
test_thevenin.py — Testes do Extrator de Equivalente Thévenin / Norton e Máxima Potência.
"""

import numpy as np
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import Resistor, VoltageSource
from circuit_solver.engine.thevenin import thevenin_norton


def test_thevenin_simple_divider():
    """Circuito:
    Vs = 20V conectado aos nós 1 e 0.
    R1 = 10Ω entre 1 e 2.
    R2 = 10Ω entre 2 e 0.
    Terminais A = "2", B = "0".

    Esperado:
    V_th = 20 * (10 / (10 + 10)) = 10V
    R_th = R1 // R2 = 10 // 10 = 5Ω
    I_N = 10V / 5Ω = 2A
    P_max = V_th^2 / (4 * R_th) = 100 / 20 = 5W
    """
    elements = [
        VoltageSource("Vs", "1", "0", 20.0),
        Resistor("R1", "1", "2", 10.0),
        Resistor("R2", "2", "0", 10.0),
    ]
    circuit = Circuit(elements)
    res = thevenin_norton(circuit, node_a="2", node_b="0")

    assert res.V_th == pytest.approx(10.0)
    assert res.R_th == pytest.approx(5.0)
    assert res.I_N == pytest.approx(2.0)
    assert res.P_max == pytest.approx(5.0)

    # Verificar que o pico da curva de varredura ocorre exatamente quando R_L ~ R_th
    max_idx = np.argmax(res.P_L_sweep)
    r_at_pmax = res.R_L_sweep[max_idx]
    assert r_at_pmax == pytest.approx(5.0, rel=0.05)
