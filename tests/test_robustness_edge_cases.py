"""
test_robustness_edge_cases.py — Suíte de Testes para Casos de Borda e Robustez Numérica.

Testes Cobertos:
----------------
1. Thévenin com impedância nula (fonte ideal direta entre terminais A e B: R_th = 0).
2. Thévenin com terminais idênticos (node_a == node_b).
3. Thévenin com circuito aberto (R_th = inf).
4. Validador com sub-rede desconectada do terra (busca em grafo BFS).
5. Validador com laço fechado de 3 fontes ideais de tensão (ciclo KVL puro).
6. Indutor em regime permanente CC (comportamento de curto-circuito).
7. Leitura de netlist SPICE com linhas de continuação '+'.
8. Acesso a tensões de nós insensível a maiúsculas/minúsculas.
9. Reprodutibilidade estatística de Monte Carlo com seed fixo.
"""

import numpy as np
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    CurrentSource,
    Inductor,
    Resistor,
    VoltageSource,
)
from circuit_solver.core.validator import CircuitValidationError
from circuit_solver.engine.mna import solve
from circuit_solver.engine.monte_carlo import monte_carlo
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.io.parser import parse_netlist_string


def test_thevenin_zero_resistance_ideal_vsource():
    """Valida extração de Thévenin para fonte ideal de tensão conectada
    diretamente entre os terminais de medição (R_th = 0 Ω, I_N = inf).
    """
    elements = [
        VoltageSource("V1", "A", "0", 12.0),
        Resistor("Rload", "A", "0", 100.0),
    ]
    circuit = Circuit(elements)
    res = thevenin_norton(circuit, node_a="A", node_b="0")

    assert res.V_th == pytest.approx(12.0)
    assert res.R_th == pytest.approx(0.0)
    assert res.I_N == float("inf")
    assert res.P_max == float("inf")
    assert len(res.P_L_sweep) == 200


def test_thevenin_same_nodes_raises():
    """Valida rejeição defensiva quando os terminais A e B são idênticos."""
    circuit = Circuit([VoltageSource("V1", "A", "0", 10.0), Resistor("R1", "A", "0", 10.0)])
    with pytest.raises(ValueError, match="não podem ser o mesmo nó"):
        thevenin_norton(circuit, node_a="A", node_b="A")


def test_thevenin_infinite_resistance_open_circuit():
    """Valida extração de Thévenin quando a porta terminal possui impedância muito alta."""
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "0", 10.0),
        # Resistor de alto isolamento conectado à porta aberta (10 MΩ)
        Resistor("Riso", "1", "open_a", 10_000_000.0),
    ]
    circuit = Circuit(elements, allowed_open_nodes=["open_a"])
    res = thevenin_norton(circuit, node_a="open_a", node_b="0")

    assert res.R_th >= 10_000_000.0
    assert res.I_N == pytest.approx(10.0 / 10_000_005.0, rel=1e-3)


def test_validator_disconnected_subcircuit_raises():
    """Valida detecção de ilhas desconectadas sem caminho para o terra."""
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "0", 10.0),
        VoltageSource("V2", "10", "11", 5.0),
        Resistor("R2", "10", "11", 100.0),
    ]
    with pytest.raises(CircuitValidationError, match="Disconnected subcircuit detected"):
        Circuit(elements)


def test_validator_voltage_source_triangle_loop_raises():
    """Valida detecção de laço fechado (ciclo KVL) com 3 fontes ideais de tensão."""
    elements = [
        VoltageSource("V1", "1", "2", 10.0),
        VoltageSource("V2", "2", "3", 5.0),
        VoltageSource("V3", "3", "1", 15.0),
        Resistor("Rgnd", "1", "0", 100.0),
    ]
    with pytest.raises(CircuitValidationError, match="Closed loop of ideal voltage sources detected"):
        Circuit(elements)


def test_mna_dc_inductor_short_circuit():
    """Valida que um indutor em regime permanente CC atua como curto-circuito ideal."""
    elements = [
        VoltageSource("Vs", "1", "0", 10.0),
        Resistor("R1", "1", "2", 5.0),
        Inductor("L1", "2", "0", 2.5),  # Em DC deve conduzir para terra com queda nula
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(10.0)
    assert sol.V("2") == pytest.approx(0.0, abs=1e-5)  # Queda praticamente nula sobre o indutor
    assert sol.I_resistor("R1") == pytest.approx(2.0, rel=1e-4)


def test_spice_parser_plus_continuation_lines():
    """Valida aglutinação correta de linhas SPICE quebradas com '+'."""
    netlist = """
* Circuito com linha de continuacao SPICE
V1 1 0 DC 24
R1 1 2
+ 6
R2 2 0
+ 12
.end
"""
    elements = parse_netlist_string(netlist)
    circuit = Circuit(elements)
    sol = solve(circuit)

    # Divisor 24V * (12 / 18) = 16.0 V
    assert sol.V("2") == pytest.approx(16.0)


def test_case_insensitive_node_lookup():
    """Valida que a consulta a nós funciona tanto em maiúsculas quanto em minúsculas."""
    elements = [
        VoltageSource("Vin", "INPUT_NODE", "0", 5.0),
        Resistor("R1", "INPUT_NODE", "0", 10.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("INPUT_NODE") == pytest.approx(5.0)
    assert sol.V("input_node") == pytest.approx(5.0)
    assert sol.V("Input_Node") == pytest.approx(5.0)


def test_monte_carlo_seed_reproducibility():
    """Valida reprodutibilidade determinística de Monte Carlo quando seed é fixo."""
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "2", 100.0, tolerance=0.10),
        Resistor("R2", "2", "0", 100.0, tolerance=0.10),
    ]
    circuit = Circuit(elements)

    res1 = monte_carlo(circuit, n_samples=200, seed=12345)
    res2 = monte_carlo(circuit, n_samples=200, seed=12345)

    np.testing.assert_allclose(res1.mean, res2.mean)
    np.testing.assert_allclose(res1.std, res2.std)
    np.testing.assert_allclose(res1.samples, res2.samples)
