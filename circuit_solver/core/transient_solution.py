"""
transient_solution.py — Solution container for time-domain transient analysis.

Wraps the time vector, node voltage waveforms V(node, t), and branch
current waveforms I(element, t) with intuitive accessors and formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from circuit_solver.core.circuit import Circuit

__all__ = ["TransientSolution"]


@dataclass
class TransientSolution:
    """Encapsulates the numerical results of a time-domain transient analysis.

    Parameters
    ----------
    circuit : Circuit
        The circuit being analyzed.
    time : np.ndarray, shape (n_steps,)
        Monotonically increasing time points in seconds.
    node_voltages : dict[str, np.ndarray]
        Mapping from non-ground node label to 1D voltage array over time.
    branch_currents : dict[str, np.ndarray]
        Mapping from element name to 1D branch current array over time.
    """

    circuit: Circuit
    time: np.ndarray
    node_voltages: dict[str, np.ndarray]
    branch_currents: dict[str, np.ndarray] = field(default_factory=dict)

    def V(self, node: str) -> np.ndarray:
        """Return the voltage waveform V(t) at ``node`` in Volts.

        Ground ('0') returns an array of zeros with the same length as ``time``.
        """
        lbl = str(node).strip()
        if lbl == "0":
            return np.zeros_like(self.time)

        if lbl in self.node_voltages:
            return self.node_voltages[lbl]

        for k, v in self.node_voltages.items():
            if k.lower() == lbl.lower():
                return v

        raise KeyError(
            f"Node '{node}' not found in transient solution. "
            f"Available nodes: {list(self.node_voltages.keys())}"
        )

    def I(self, element_name: str) -> np.ndarray:
        """Return the current waveform I(t) through ``element_name`` in Amperes."""
        name = str(element_name).strip()
        if name in self.branch_currents:
            return self.branch_currents[name]

        for k, v in self.branch_currents.items():
            if k.lower() == name.lower():
                return v

        raise KeyError(
            f"Branch current for '{element_name}' not recorded. "
            f"Available elements: {list(self.branch_currents.keys())}"
        )

    def P(self, element_name: str) -> np.ndarray:
        """Return the instantaneous power waveform P(t) absorbed by ``element_name`` in Watts."""
        el = None
        for item in self.circuit.elements:
            if item.name.lower() == element_name.lower():
                el = item
                break

        if el is None:
            raise KeyError(f"Element '{element_name}' not found in circuit.")

        # For 2-terminal elements: P(t) = (V_p(t) - V_n(t)) * I(t)
        if hasattr(el, "node_p") and hasattr(el, "node_n"):
            v_p = self.V(el.node_p)
            v_n = self.V(el.node_n)
            v_branch = v_p - v_n
            i_branch = self.I(el.name)
            return v_branch * i_branch

        raise NotImplementedError(f"Power calculation not implemented for multi-terminal element '{element_name}'.")

    @property
    def n_steps(self) -> int:
        """Total number of time simulation steps."""
        return len(self.time)

    @property
    def t_stop(self) -> float:
        """Final simulation time in seconds."""
        return float(self.time[-1]) if len(self.time) > 0 else 0.0

    def summary(self) -> str:
        """Format a concise overview of the transient run."""
        lines = [
            "=" * 72,
            "  Transient Analysis Summary",
            "=" * 72,
            f"  Time span: 0.0 s -> {self.t_stop:.6e} s  ({self.n_steps} steps)",
            f"  Nodes tracked: {len(self.node_voltages)}",
            "-" * 72,
            f"  {'Node':>8}  {'V_final (V)':>14}  {'V_min (V)':>12}  {'V_max (V)':>12}",
            "-" * 72,
        ]
        for node, v_arr in self.node_voltages.items():
            lines.append(
                f"  {node:>8}  {v_arr[-1]:>14.6f}  {np.min(v_arr):>12.6f}  {np.max(v_arr):>12.6f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)
