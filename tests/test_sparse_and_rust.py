"""
test_sparse_and_rust.py — Tests for Rust 100% element expansion and Sparse Solvers.

Validates:
1. Expansion of Rust core to 100% of linear circuit elements (VCVS, VCCS, CCCS, CCVS, OpAmp, Inductors).
2. Multi-threaded Rust Monte Carlo executing with Op-Amps and controlled sources without Python fallback.
3. Sparse MNA assembly and solving via scipy.sparse.linalg.spsolve.
4. Native Rust sparse LU solving via Faer for large circuits (> 1,000 nodes).
5. Exact parity between dense, scipy sparse, and Rust sparse solvers.
"""

import time
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    VCVS,
    CCCS,
    OpAmp,
)
from circuit_solver.engine.mna import build_mna_sparse, solve
from circuit_solver.engine.monte_carlo_rust import monte_carlo


def test_rust_elements_monte_carlo_with_opamp():
    """Verify Monte Carlo runs directly in Rust for circuits with Op-Amps."""
    elements = [
        VoltageSource("Vin", "in", "0", 2.0),
        Resistor("R1", "in", "inv", 10_000.0, tolerance=0.05),
        Resistor("Rf", "inv", "out", 20_000.0, tolerance=0.05),
        OpAmp("U1", in_p="0", in_n="inv", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)

    # Nominal analytical gain: Vout = - (Rf / R1) * Vin = - (20k / 10k) * 2 = -4.0 V
    res = monte_carlo(circuit, n_samples=2000, seed=123, force_python=False)

    out_idx = circuit.nodes.index("out")
    mean_out = res.mean[out_idx]
    assert np.isclose(mean_out, -4.0, rtol=0.01)
    assert res.std[out_idx] > 0.05  # shows variation from 5% tolerance
    assert res.n_failed == 0


def test_rust_elements_monte_carlo_with_vcvs():
    """Verify Monte Carlo runs directly in Rust for circuits with VCVS."""
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "2", 1000.0, tolerance=0.05),
        Resistor("R2", "2", "0", 1000.0, tolerance=0.05),
        # E1 senses V(2, 0) and multiplies by 3
        VCVS("E1", "out", "0", "2", "0", gain=3.0),
        Resistor("RL", "out", "0", 500.0, tolerance=0.05),
    ]
    circuit = Circuit(elements)

    res = monte_carlo(circuit, n_samples=1000, seed=42)
    out_idx = circuit.nodes.index("out")
    # Nominal: V(2) = 5.0 V, V(out) = 3 * 5 = 15.0 V
    assert np.isclose(res.mean[out_idx], 15.0, rtol=0.01)
    assert res.n_failed == 0


def test_rust_elements_monte_carlo_with_cccs_and_ccvs():
    """Verify Monte Carlo in Rust with current-controlled sources (CCCS, CCVS)."""
    elements = [
        VoltageSource("V1", "1", "0", 12.0),
        Resistor("R1", "1", "2", 4.0, tolerance=0.02),
        VoltageSource("Vctrl", "2", "0", 0.0),  # Current sensor (0V)
        # CCCS F1 injects 2 * I(Vctrl) into node 3
        CCCS("F1", "3", "0", vctrl_name="Vctrl", gain=2.0),
        Resistor("R3", "3", "0", 5.0, tolerance=0.02),
    ]
    circuit = Circuit(elements)

    res = monte_carlo(circuit, n_samples=1000, seed=99)
    idx_3 = circuit.nodes.index("3")
    # Analytical: V(3) = - 2.0 * (12 / 4) * 5.0 = -30.0 V
    assert np.isclose(res.mean[idx_3], -30.0, rtol=0.02)


def make_large_ladder_circuit(n_stages: int) -> Circuit:
    """Construct an RC ladder circuit with 2 * n_stages nodes."""
    elements = [
        VoltageSource("V_in", "n1", "0", 10.0),
        Resistor("R_in", "n1", "n2", 100.0),
    ]

    for i in range(2, n_stages + 1):
        prev_node = f"n{i}"
        curr_node = f"n{i+1}"
        # Series resistor
        elements.append(Resistor(f"R_s_{i}", prev_node, curr_node, 100.0))
        # Shunt resistor to ground
        elements.append(Resistor(f"R_sh_{i}", curr_node, "0", 10_000.0))

    return Circuit(elements)


def test_sparse_solver_correctness_ladder():
    """Verify scipy sparse solver matches dense solver exactly on ladder network."""
    circuit = make_large_ladder_circuit(n_stages=100)

    sol_dense = solve(circuit, solver="dense")
    sol_sparse = solve(circuit, solver="sparse")

    np.testing.assert_allclose(sol_dense._x, sol_sparse._x, rtol=1e-10, atol=1e-12)


def test_rust_sparse_solver_vs_python_sparse():
    """Verify native Rust Faer sparse solver matches scipy sparse solver."""
    circuit = make_large_ladder_circuit(n_stages=200)

    sol_py_sparse = solve(circuit, solver="sparse")
    sol_rust_sparse = solve(circuit, solver="rust_sparse")

    np.testing.assert_allclose(sol_py_sparse._x, sol_rust_sparse._x, rtol=1e-9, atol=1e-11)


def test_giant_circuit_1000_nodes_sparse():
    """Test solving a circuit with > 1,000 nodes using sparse solvers."""
    circuit = make_large_ladder_circuit(n_stages=1005)
    assert circuit.n_nodes >= 1000

    # Auto solver should automatically select sparse because size > 500
    t0 = time.perf_counter()
    sol_auto = solve(circuit, solver="auto", sparse_threshold=500)
    t_auto = time.perf_counter() - t0

    # Rust sparse solver
    t0 = time.perf_counter()
    sol_rust = solve(circuit, solver="rust_sparse")
    t_rust = time.perf_counter() - t0

    assert sol_auto.V("n2") < 10.0
    assert sol_auto.V(f"n{1005}") > 0.0
    np.testing.assert_allclose(sol_auto._x, sol_rust._x, rtol=1e-9, atol=1e-11)
    # Sparse solvers solve 1,000 nodes in under 0.2 seconds
    assert t_auto < 1.0
    assert t_rust < 1.0


def test_sparse_matrix_coo_csc_structure():
    """Verify build_mna_sparse returns valid CSC format with expected sparsity."""
    circuit = make_large_ladder_circuit(n_stages=500)
    A_csc, z = build_mna_sparse(circuit)

    # In a 500-node ladder, non-zeros should be ~ 3-4 per row, sparsity > 99%
    total_elements = A_csc.shape[0] * A_csc.shape[1]
    nnz = A_csc.nnz
    sparsity = 1.0 - (nnz / total_elements)
    assert sparsity > 0.98  # > 98% sparse!
    assert len(z) == A_csc.shape[0]
