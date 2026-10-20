"""
solution.py — Post-processing of the raw MNA solution vector.

:class:`Solution` wraps the raw numpy array returned by the MNA solver
and provides human-readable accessors for:

* Node voltages (``V``).
* Branch currents — through voltage sources, resistors, current sources.
* Element power dissipation / generation.
* Tellegen power audit (∑P = 0).

All currents follow the **passive sign convention** unless otherwise noted.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import numpy as np

from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    CCCS,
    CCVS,
    Capacitor,
    Inductor,
    OpAmp,
)

if TYPE_CHECKING:
    from circuit_solver.core.circuit import Circuit

__all__ = ["Solution"]


class Solution:
    """Wraps the MNA solution vector with labelled accessors.

    Parameters
    ----------
    circuit : Circuit
        The circuit description used to build the MNA system.
    x : np.ndarray
        Raw solution vector of length ``circuit.size``.
    """

    def __init__(self, circuit: "Circuit", x: np.ndarray) -> None:
        self._circuit = circuit
        self._x = x

    # ------------------------------------------------------------------
    # Node voltages
    # ------------------------------------------------------------------

    @property
    def voltages(self) -> dict[str, float]:
        """Mapping from node label (including '0') to voltage in Volts."""
        result: dict[str, float] = {"0": 0.0}
        for label, idx in self._circuit.node_map.items():
            result[label] = float(self._x[idx])
        return result

    def V(self, node: str) -> float:
        """Voltage at a single node.

        Parameters
        ----------
        node : str
            Node label.  ``'0'`` always returns 0.0.
        """
        if str(node).strip() == "0":
            return 0.0
        n_str = str(node).strip()
        if n_str in self._circuit.node_map:
            return float(self._x[self._circuit.node_map[n_str]])
        for k, idx in self._circuit.node_map.items():
            if k.lower() == n_str.lower():
                return float(self._x[idx])
        raise KeyError(f"Node '{node}' is not in circuit.")

    def V_diff(self, node_p: str, node_n: str) -> float:
        """Differential voltage ``V(node_p) − V(node_n)``."""
        return self.V(node_p) - self.V(node_n)

    # ------------------------------------------------------------------
    # Branch currents
    # ------------------------------------------------------------------

    def I_vsource(self, name: str) -> float:
        """Current *through* a voltage source or VCVS (from + to − internally).

        The auxiliary variable ``j`` is the current *into* the positive terminal
        from the rest of the circuit (conventional current exits the + terminal
        of the source — generator convention).  We negate to return current in
        the direction of voltage rise (load convention).
        """
        idx = self._circuit.aux_index(name)
        return float(self._x[idx])

    def I_resistor(self, name: str) -> float:
        """Current through a resistor (from node_p to node_n)."""
        el = self._element_by_name(name, Resistor)
        vp = self.V(el.node_p)
        vn = self.V(el.node_n)
        return (vp - vn) / el.resistance

    def I_isource(self, name: str) -> float:
        """Nominal current of an independent current source (always returns
        the set-point; actual KCL is enforced by the solver)."""
        el = self._element_by_name(name, CurrentSource)
        return el.current

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def P_resistor(self, name: str) -> float:
        """Power *dissipated* by a resistor (always ≥ 0 for R > 0)."""
        el = self._element_by_name(name, Resistor)
        v = self.V_diff(el.node_p, el.node_n)
        return v ** 2 / el.resistance

    def P_vsource(self, name: str) -> float:
        """Power *delivered* by a voltage source (positive = source supplies power)."""
        el = self._element_by_name(name, VoltageSource)
        v = self.V_diff(el.node_p, el.node_n)
        j = self.I_vsource(name)
        # j flows into the + terminal from outside; source delivers P = v * j
        return v * (-j)

    def P_isource(self, name: str) -> float:
        """Power *delivered* by a current source."""
        el = self._element_by_name(name, CurrentSource)
        v = self.V_diff(el.node_p, el.node_n)
        return v * el.current

    # ------------------------------------------------------------------
    # Tellegen audit
    # ------------------------------------------------------------------

    def power_audit(self) -> dict[str, float]:
        """Compute power for every element and check Tellegen's theorem.

        Returns
        -------
        dict[str, float]
            ``{"element_name": power, ..., "_total": sum}``

        Raises
        ------
        RuntimeWarning
            If |∑P| > 1 µW (numerical imbalance likely indicates a bug).
        """
        powers: dict[str, float] = {}
        for el in self._circuit.elements:
            if isinstance(el, Resistor):
                powers[el.name] = -self.P_resistor(el.name)  # negative = absorbed
            elif isinstance(el, VoltageSource):
                powers[el.name] = self.P_vsource(el.name)
            elif isinstance(el, CurrentSource):
                powers[el.name] = self.P_isource(el.name)
            elif isinstance(el, VCVS):
                v = self.V_diff(el.node_p, el.node_n)
                j = self.I_vsource(el.name)
                powers[el.name] = v * (-j)
            elif isinstance(el, VCCS):
                v = self.V_diff(el.node_p, el.node_n)
                v_ctrl = self.V_diff(el.ctrl_p, el.ctrl_n)
                i_out = el.transconductance * v_ctrl
                powers[el.name] = v * i_out
            elif isinstance(el, CCVS):
                v = self.V_diff(el.node_p, el.node_n)
                j = self.I_vsource(el.name)
                powers[el.name] = v * (-j)
            elif isinstance(el, CCCS):
                v = self.V_diff(el.node_p, el.node_n)
                i_ctrl = -self.I_vsource(el.vctrl_name)
                i_out = el.gain * i_ctrl
                powers[el.name] = v * i_out
            elif isinstance(el, OpAmp):
                # Op-amp output supplies current; input draws ~0 (infinite Rin)
                v_out = self.V(el.out)
                j = self.I_vsource(el.name)
                powers[el.name] = v_out * (-j)

        total = sum(powers.values())
        powers["_total"] = total

        if abs(total) > 1e-6:
            import warnings
            warnings.warn(
                f"Tellegen power imbalance: Σ P = {total:.3e} W. "
                "Check circuit for numerical issues.",
                RuntimeWarning,
                stacklevel=2,
            )
        return powers

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted human-readable solution summary."""
        lines = ["=" * 56, "  pycircuit-solver — MNA Solution", "=" * 56]

        lines.append("\n  Node Voltages")
        lines.append("  " + "-" * 30)
        for label, v in sorted(self.voltages.items()):
            lines.append(f"  V({label:>6}) = {v:>12.6f} V")

        lines.append("\n  Branch Currents (Voltage Sources / VCVS / CCVS / Op-Amps)")
        lines.append("  " + "-" * 30)
        for el in self._circuit.elements:
            if isinstance(el, (VoltageSource, VCVS, CCVS, OpAmp)):
                j = self.I_vsource(el.name)
                lines.append(f"  I({el.name:>6}) = {j:>12.6f} A")

        lines.append("\n  Resistor Currents")
        lines.append("  " + "-" * 30)
        for el in self._circuit.elements:
            if isinstance(el, Resistor):
                i = self.I_resistor(el.name)
                lines.append(f"  I({el.name:>6}) = {i:>12.6f} A")

        lines.append("\n  Power (positive = source delivers, negative = load absorbs)")
        lines.append("  " + "-" * 30)
        audit = self.power_audit()
        for k, p in audit.items():
            if k == "_total":
                continue
            lines.append(f"  P({k:>6}) = {p:>12.6f} W")
        lines.append("  " + "-" * 30)
        lines.append(f"  Σ P (Tellegen) = {audit['_total']:>12.6e} W")
        lines.append("=" * 56)
        return "\n".join(lines)

    def __repr__(self) -> str:
        v = ", ".join(f"V({k})={val:.4f}" for k, val in self.voltages.items() if k != "0")
        return f"Solution({v})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _element_by_name(self, name: str, kind: type) -> object:
        for el in self._circuit.elements:
            if el.name == name and isinstance(el, kind):
                return el
        raise KeyError(f"No {kind.__name__} named '{name}' in circuit.")
