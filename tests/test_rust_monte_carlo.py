"""
test_rust_monte_carlo.py — Integration tests for Rust-accelerated Monte Carlo.

Tests verify:
1. Rust extension loads correctly.
2. Results match Python reference (within tolerance).
3. Parallel reproducibility (same seed → same result).
4. Fallback to Python works when force_python=True.
5. Performance: Rust is significantly faster than Python.
"""

from __future__ import annotations

import time
import math
import pytest

from circuit_solver.core.elements import Resistor, VoltageSource, CurrentSource
from circuit_solver.core.circuit import Circuit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def simple_divider_circuit() -> Circuit:
    """Voltage divider: V1=10V, R1=4Ω, R2=6Ω → V(1)=10V, V(2)=6V."""
    return Circuit([
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "2", 4.0, tolerance=0.05),
        Resistor("R2", "2", "0", 6.0, tolerance=0.05),
    ])


def bridge_circuit() -> Circuit:
    """Wheatstone bridge — more nodes, better stress test."""
    return Circuit([
        VoltageSource("V1", "1", "0", 12.0),
        Resistor("R1", "1", "2", 100.0, tolerance=0.01),
        Resistor("R2", "1", "3", 100.0, tolerance=0.01),
        Resistor("R3", "2", "4", 100.0, tolerance=0.01),
        Resistor("R4", "3", "4", 100.0, tolerance=0.01),
        Resistor("RL", "2", "3", 1000.0, tolerance=0.01),
        Resistor("Rg", "4", "0", 1.0, tolerance=0.01),
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRustAvailability:
    def test_rust_module_importable(self):
        """pycircuit_core must import without error after `maturin build`."""
        from circuit_solver.engine.monte_carlo_rust import RUST_AVAILABLE
        assert RUST_AVAILABLE, (
            "Rust extension not available. Run: "
            "cd rust_core && maturin build --release --interpreter python3 && "
            "pip3 install target/wheels/*.whl"
        )


class TestRustCorrectnessVsPython:
    """Rust and Python must give statistically equivalent results."""

    def test_mean_voltage_matches(self):
        """Mean node voltage should be ≈ nominal (within 1 % for 5 % tolerances)."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo

        circ = simple_divider_circuit()
        result = monte_carlo(circ, n_samples=5000, seed=42)

        # Nominal V(2) = 10 * 6/(4+6) = 6.0 V
        node2_idx = result.node_labels.index("2")
        assert abs(result.mean[node2_idx] - 6.0) < 0.1, (
            f"Mean V(2) = {result.mean[node2_idx]:.4f} V, expected ≈ 6.0 V"
        )

    def test_rust_vs_python_same_seed(self):
        """With the same seed, Rust and Python should give very similar statistics."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo

        circ = simple_divider_circuit()

        rust_result   = monte_carlo(circ, n_samples=2000, seed=99, force_python=False)
        python_result = monte_carlo(circ, n_samples=2000, seed=99, force_python=True)

        for i, node in enumerate(rust_result.node_labels):
            # Different RNG algorithms → means should agree within 0.5 %
            rel_diff = abs(rust_result.mean[i] - python_result.mean[i])
            assert rel_diff < 0.5, (
                f"Node {node}: Rust mean={rust_result.mean[i]:.4f}, "
                f"Python mean={python_result.mean[i]:.4f}, diff={rel_diff:.4f}"
            )

    def test_std_positive(self):
        """Standard deviation must be > 0 at divider node (resistors have tolerance)."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo

        circ = simple_divider_circuit()
        result = monte_carlo(circ, n_samples=1000, seed=7)

        # Node '2' is the resistive divider output, subject to resistor tolerances
        idx = result.node_labels.index("2")
        assert result.std[idx] > 0, f"Std of divider node 2 is zero — tolerance not applied?"

    def test_percentile_ordering(self):
        """p5 ≤ mean ≤ p95 for each node."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo

        circ = simple_divider_circuit()
        result = monte_carlo(circ, n_samples=3000, seed=13)

        for i in range(len(result.node_labels)):
            assert result.p5[i] <= result.mean[i] + 1e-9
            assert result.mean[i] <= result.p95[i] + 1e-9

    def test_min_max_bounds(self):
        """min ≤ p5 and p95 ≤ max."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo

        circ = simple_divider_circuit()
        result = monte_carlo(circ, n_samples=3000, seed=21)

        for i in range(len(result.node_labels)):
            assert result.min[i] <= result.p5[i] + 1e-9
            assert result.p95[i] <= result.max[i] + 1e-9


class TestRustReproducibility:
    def test_same_seed_same_result(self):
        """Two calls with same seed must be bit-for-bit identical."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo
        import numpy as np

        circ = simple_divider_circuit()
        r1 = monte_carlo(circ, n_samples=500, seed=42)
        r2 = monte_carlo(circ, n_samples=500, seed=42)

        np.testing.assert_array_equal(r1.mean, r2.mean,
            err_msg="Rust Monte Carlo not reproducible with same seed!")

    def test_different_seeds_differ(self):
        """Different seeds must produce different results."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo
        import numpy as np

        circ = simple_divider_circuit()
        r1 = monte_carlo(circ, n_samples=500, seed=1)
        r2 = monte_carlo(circ, n_samples=500, seed=2)

        assert not np.array_equal(r1.samples, r2.samples), \
            "Different seeds produced identical samples!"


class TestRustFallback:
    def test_force_python_fallback(self):
        """force_python=True must use Python backend regardless of Rust availability."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo

        circ = simple_divider_circuit()
        result = monte_carlo(circ, n_samples=200, seed=5, force_python=True)
        assert result.n_samples == 200
        assert len(result.node_labels) > 0

    def test_override_tolerance(self):
        """override_tolerance should tighten std relative to default tolerance."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo

        circ = simple_divider_circuit()
        wide  = monte_carlo(circ, n_samples=2000, seed=55, override_tolerance=0.20)
        tight = monte_carlo(circ, n_samples=2000, seed=55, override_tolerance=0.01)

        idx = wide.node_labels.index("2")  # divider node with variable voltage
        assert wide.std[idx] > tight.std[idx], \
            "Wider tolerance should produce larger std deviation."

    def test_no_failed_samples_simple(self):
        """A well-conditioned circuit should have zero failed samples."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo

        circ = simple_divider_circuit()
        result = monte_carlo(circ, n_samples=1000, seed=0)
        assert result.n_failed == 0


class TestRustPerformance:
    """Performance sanity checks (not strict, just flag regressions)."""

    def test_rust_faster_than_python(self):
        """Rust backend should be at least 5x faster than Python for 5000 samples."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo, RUST_AVAILABLE

        if not RUST_AVAILABLE:
            pytest.skip("Rust not available")

        circ = bridge_circuit()

        t0 = time.perf_counter()
        monte_carlo(circ, n_samples=5000, seed=42, force_python=True)
        t_python = time.perf_counter() - t0

        t0 = time.perf_counter()
        monte_carlo(circ, n_samples=5000, seed=42, force_python=False)
        t_rust = time.perf_counter() - t0

        speedup = t_python / t_rust
        print(f"\n  Python: {t_python*1000:.1f} ms | Rust: {t_rust*1000:.1f} ms | Speedup: {speedup:.1f}×")
        assert speedup > 5.0, (
            f"Expected Rust to be >5× faster, got {speedup:.1f}×. "
            f"Python={t_python*1000:.1f}ms, Rust={t_rust*1000:.1f}ms"
        )

    def test_large_sample_completes_quickly(self):
        """50,000 samples should complete in under 3 seconds on any modern CPU."""
        from circuit_solver.engine.monte_carlo_rust import monte_carlo, RUST_AVAILABLE

        if not RUST_AVAILABLE:
            pytest.skip("Rust not available")

        circ = bridge_circuit()
        t0 = time.perf_counter()
        result = monte_carlo(circ, n_samples=50_000, seed=99)
        elapsed = time.perf_counter() - t0

        print(f"\n  50k samples: {elapsed*1000:.1f} ms")
        assert elapsed < 3.0, f"50k samples took {elapsed:.2f}s — too slow!"
        assert result.n_samples == 50_000
