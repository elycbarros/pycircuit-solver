"""
monte_carlo_rust.py — Python adapter that bridges Circuit → Rust MNA core.

This module resolves node names to integer indices, builds the Rust-compatible
descriptor objects (PyResistor, PyVSource, PyISource), and calls the Rust
`monte_carlo_run` function via PyO3.

Fallback
--------
If the Rust extension `pycircuit_core` is not available (e.g. not compiled
yet, unsupported platform), the module transparently falls back to the
pure-Python implementation in `circuit_solver.engine.monte_carlo`.

Usage (identical API to the Python version)
-------------------------------------------
>>> from circuit_solver.engine.monte_carlo_rust import monte_carlo
>>> result = monte_carlo(circuit, n_samples=10_000)
>>> print(result.summary())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    _Element,
)
from circuit_solver.engine.monte_carlo import MonteCarloResult

# ---------------------------------------------------------------------------
# Try to import the Rust extension
# ---------------------------------------------------------------------------
try:
    import pycircuit_core as _rust  # type: ignore[import]
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

__all__ = ["monte_carlo", "RUST_AVAILABLE"]

RUST_AVAILABLE: bool = _RUST_AVAILABLE


# ---------------------------------------------------------------------------
# Circuit → Rust descriptor builder
# ---------------------------------------------------------------------------

def _build_rust_descriptors(circuit: Circuit) -> tuple:
    """Convert a Circuit object into Rust-compatible descriptor lists.

    Returns
    -------
    n_nodes, size, resistors, vsources, isources
    """
    node_idx = circuit.node_index      # str → int | None
    aux_idx  = circuit.aux_index       # str → int

    py_resistors: list = []
    py_vsources:  list = []
    py_isources:  list = []

    for el in circuit.elements:
        if isinstance(el, Resistor):
            p  = node_idx(el.node_p)
            n  = node_idx(el.node_n)
            py_resistors.append(
                _rust.PyResistor(
                    el.resistance,
                    el.tolerance,
                    p  if p  is not None else -1,
                    n  if n  is not None else -1,
                )
            )

        elif isinstance(el, VoltageSource):
            p   = node_idx(el.node_p)
            n   = node_idx(el.node_n)
            aux = aux_idx(el.name)
            py_vsources.append(
                _rust.PyVSource(
                    el.voltage,
                    p   if p   is not None else -1,
                    n   if n   is not None else -1,
                    aux,
                )
            )

        elif isinstance(el, CurrentSource):
            p = node_idx(el.node_p)
            n = node_idx(el.node_n)
            py_isources.append(
                _rust.PyISource(
                    el.current,
                    p if p is not None else -1,
                    n if n is not None else -1,
                )
            )
        # Dependent sources / OpAmps are left to the Python fallback for now.

    return (
        circuit.n_nodes,  # non-ground nodes
        circuit.size,     # MNA system size
        py_resistors,
        py_vsources,
        py_isources,
    )


def _has_unsupported_elements(circuit: Circuit) -> bool:
    """Return True if the circuit contains elements not yet in the Rust core."""
    from circuit_solver.core.elements import VCVS, VCCS, CCCS, CCVS, OpAmp
    for el in circuit.elements:
        if isinstance(el, (VCVS, VCCS, CCCS, CCVS, OpAmp)):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def monte_carlo(
    circuit: Circuit,
    *,
    n_samples: int = 1000,
    seed: int | None = 42,
    override_tolerance: float | None = None,
    force_python: bool = False,
) -> MonteCarloResult:
    """Run Monte Carlo simulation — Rust-accelerated when available.

    Parameters
    ----------
    circuit : Circuit
        The nominal circuit. Resistor tolerances are taken from each
        element's ``.tolerance`` attribute (default 5 %).
    n_samples : int
        Number of random simulation runs.
    seed : int | None
        Random seed for reproducibility. Pass ``None`` for non-deterministic.
    override_tolerance : float | None
        If provided, overrides every resistor's tolerance with this value.
    force_python : bool
        If True, skip the Rust backend even if available (useful for testing).

    Returns
    -------
    MonteCarloResult
        Statistical summary of node voltages (same object as Python version).
    """
    use_rust = (
        _RUST_AVAILABLE
        and not force_python
        and not _has_unsupported_elements(circuit)
    )

    if use_rust:
        return _monte_carlo_rust(circuit, n_samples=n_samples, seed=seed,
                                  override_tolerance=override_tolerance)
    else:
        if _RUST_AVAILABLE and _has_unsupported_elements(circuit):
            warnings.warn(
                "Circuit contains dependent sources / Op-Amps not yet "
                "supported by the Rust core. Falling back to Python.",
                stacklevel=2,
            )
        from circuit_solver.engine.monte_carlo import monte_carlo as _py_mc
        return _py_mc(circuit, n_samples=n_samples, seed=seed,
                      override_tolerance=override_tolerance)


def _monte_carlo_rust(
    circuit: Circuit,
    *,
    n_samples: int,
    seed: int | None,
    override_tolerance: float | None,
) -> MonteCarloResult:
    """Internal: call the Rust Monte Carlo kernel."""
    n_nodes, size, resistors, vsources, isources = _build_rust_descriptors(circuit)

    rust_seed = int(seed) if seed is not None else 0

    mean, std, mn, mx, p5, p95, samples_2d, n_failed = _rust.monte_carlo_run(
        n_nodes,
        size,
        resistors,
        vsources,
        isources,
        n_samples,
        rust_seed,
        override_tolerance,
    )

    # samples_2d is already a numpy array (zero-copy from Rust)
    return MonteCarloResult(
        node_labels=list(circuit.nodes),
        samples=samples_2d,
        mean=mean,
        std=std,
        min=mn,
        max=mx,
        p5=p5,
        p95=p95,
        n_samples=n_samples,
        n_failed=n_failed,
    )
