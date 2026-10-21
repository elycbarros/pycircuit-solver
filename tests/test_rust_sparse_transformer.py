import pytest

from circuit_solver import Circuit, solve
from circuit_solver.core.elements import Resistor, Transformer, VoltageSource
from circuit_solver.engine.monte_carlo_rust import RUST_AVAILABLE


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust não compilado nesta máquina")
def test_rust_sparse_keeps_transformer_in_dc():
    circ = Circuit([
        VoltageSource("V1", "1", "0", 1.0),
        Resistor("R1", "1", "2", 1e3),
        Transformer("T1", "2", "0", "3", "0", L1=1.0, L2=1.0, coupling=0.999),
        Resistor("R2", "3", "0", 1e3),
    ])
    ref = solve(circ, solver="sparse")
    got = solve(circ, solver="rust_sparse")
    assert got.V("2") == pytest.approx(ref.V("2"), abs=1e-6)
    assert got.V("3") == pytest.approx(ref.V("3"), abs=1e-6)
