"""
monte_carlo.py — Statistical worst-case analysis via Monte Carlo simulation.

For each resistor in the circuit, the actual resistance is drawn from a
uniform distribution centred on the nominal value within ±tolerance:

    R_actual ~ Uniform(R_nominal*(1-tol), R_nominal*(1+tol))

The solver is run ``n_samples`` times and statistics (mean, std, min/max,
percentiles) are computed for every node voltage and selected branch quantities.

Typical commercial tolerances
------------------------------
* ±1 %   → ``tolerance=0.01``
* ±5 %   → ``tolerance=0.05``  (default)
* ±10 %  → ``tolerance=0.10``

Usage
-----
>>> result = monte_carlo(circuit, n_samples=1000)
>>> result.summary()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import Resistor, _Element
from circuit_solver.engine.mna import solve

__all__ = ["MonteCarloResult", "monte_carlo"]


@dataclass
class MonteCarloResult:
    """Statistical summary of Monte Carlo node-voltage distributions.

    Attributes
    ----------
    node_labels : list[str]
        Ordered list of non-ground node labels.
    samples : np.ndarray, shape (n_samples, n_nodes)
        All sampled node-voltage vectors.
    mean : np.ndarray, shape (n_nodes,)
    std : np.ndarray, shape (n_nodes,)
    min : np.ndarray, shape (n_nodes,)
    max : np.ndarray, shape (n_nodes,)
    p5 : np.ndarray — 5th percentile
    p95 : np.ndarray — 95th percentile
    n_samples : int
    n_failed : int
        Number of samples that failed to solve (singular matrix, etc.).
    """

    node_labels: list[str]
    samples: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    p5: np.ndarray
    p95: np.ndarray
    n_samples: int
    n_failed: int

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"  Monte Carlo Analysis  ({self.n_samples} samples, "
            f"{self.n_failed} failed)",
            "=" * 72,
            f"  {'Node':>6}  {'Mean (V)':>12}  {'Std (V)':>10}  "
            f"{'Min (V)':>10}  {'Max (V)':>10}  {'P5 (V)':>10}  {'P95 (V)':>10}",
            "  " + "-" * 68,
        ]
        for i, label in enumerate(self.node_labels):
            lines.append(
                f"  {label:>6}  {self.mean[i]:>12.6f}  {self.std[i]:>10.6f}  "
                f"{self.min[i]:>10.6f}  {self.max[i]:>10.6f}  "
                f"{self.p5[i]:>10.6f}  {self.p95[i]:>10.6f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)


def monte_carlo(
    circuit: Circuit,
    *,
    n_samples: int = 1000,
    seed: int | None = 42,
    override_tolerance: float | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on resistor tolerances.

    Parameters
    ----------
    circuit : Circuit
        The nominal circuit.  Resistor tolerances are taken from each
        element's ``.tolerance`` attribute (default 5 %).
    n_samples : int
        Number of random simulation runs.
    seed : int | None
        Random seed for reproducibility.  Pass ``None`` for non-deterministic.
    override_tolerance : float | None
        If provided, overrides every resistor's tolerance with this value.

    Returns
    -------
    MonteCarloResult
        Statistical summary of node voltages across all runs.
    """
    rng = np.random.default_rng(seed)
    node_labels = circuit.nodes  # ordered non-ground nodes

    all_voltages: list[np.ndarray] = []
    n_failed = 0

    for _ in range(n_samples):
        # Perturb resistor values
        perturbed: list[_Element] = []
        for el in circuit.elements:
            if isinstance(el, Resistor):
                tol = override_tolerance if override_tolerance is not None else el.tolerance
                delta = rng.uniform(-tol, tol)
                r_actual = el.resistance * (1.0 + delta)
                # Clamp to a physically sensible minimum
                r_actual = max(r_actual, 1e-9)
                perturbed.append(replace(el, resistance=r_actual))
            else:
                perturbed.append(el)

        try:
            circ = Circuit(perturbed, validate=False)
            sol = solve(circ, check_singular=False)
            v_vec = np.array([sol.V(n) for n in node_labels])
            all_voltages.append(v_vec)
        except Exception:
            n_failed += 1

    if not all_voltages:
        raise RuntimeError(
            "All Monte Carlo samples failed to solve. "
            "Check that the nominal circuit is well-conditioned."
        )

    samples = np.array(all_voltages)  # (n_valid, n_nodes)

    return MonteCarloResult(
        node_labels=node_labels,
        samples=samples,
        mean=np.mean(samples, axis=0),
        std=np.std(samples, axis=0),
        min=np.min(samples, axis=0),
        max=np.max(samples, axis=0),
        p5=np.percentile(samples, 5, axis=0),
        p95=np.percentile(samples, 95, axis=0),
        n_samples=n_samples,
        n_failed=n_failed,
    )
