"""
dc_sweep.py — DC parametric sweep engine (.dc sweep).

Supports:
- Sweeping a DC independent voltage source (V), current source (I), or resistor (R).
- Single sweep (e.g. V_in from 0 to 10 V with step 0.1 V).
- Nested / 2D sweep (e.g. sweeping V_CE across multiple base currents I_B to generate
  transistor characteristic curve families I_C x V_CE).
- Transfer curves (V_out x V_in), load curves, and component I-V characteristics.
- Automatic dispatch to linear or non-linear (Newton-Raphson) solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    Inductor,
    Transformer,
    VoltageSource,
    CurrentSource,
    Diode,
    ZenerDiode,
    BJT,
    MOSFET,
)
from circuit_solver.engine.mna import solve


@dataclass
class DCSweepResult:
    """Container for DC parametric sweep simulation results.

    Parameters
    ----------
    param_name : str
        Name of primary swept component/parameter (e.g. 'V1', 'Rload').
    sweep_values : np.ndarray
        Array of primary swept values.
    node_voltages : dict[str, np.ndarray]
        Mapping from node label to numpy array of voltages.
    branch_currents : dict[str, np.ndarray]
        Mapping from element name to numpy array of branch currents.
    param2_name : str | None, optional
        Name of secondary swept component if nested sweep was performed.
    sweep2_values : np.ndarray | None, optional
        Array of secondary swept values if nested sweep was performed.
    """

    param_name: str
    sweep_values: np.ndarray
    node_voltages: dict[str, np.ndarray]
    branch_currents: dict[str, np.ndarray]
    param2_name: str | None = None
    sweep2_values: np.ndarray | None = None

    def V(self, node: str) -> np.ndarray:
        """Return voltage array at ``node`` across the sweep."""
        n_str = str(node).strip()
        if n_str == "0":
            shape = (len(self.sweep2_values), len(self.sweep_values)) if self.sweep2_values is not None else (len(self.sweep_values),)
            return np.zeros(shape, dtype=float)
        if n_str in self.node_voltages:
            return self.node_voltages[n_str]
        for k, arr in self.node_voltages.items():
            if k.lower() == n_str.lower():
                return arr
        raise KeyError(f"Node '{node}' was not found in DC sweep results.")

    def I(self, element: str) -> np.ndarray:
        """Return branch current array for ``element`` across the sweep."""
        el_str = str(element).strip()
        if el_str in self.branch_currents:
            return self.branch_currents[el_str]
        for k, arr in self.branch_currents.items():
            if k.lower() == el_str.lower():
                return arr
        raise KeyError(f"Element '{element}' branch current was not found in DC sweep results.")

    def summary(self) -> str:
        """Format a human-readable text summary of the DC sweep."""
        lines = [
            "=" * 60,
            "  DC Parametric Sweep Summary",
            "=" * 60,
            f"  Primary parameter   : {self.param_name}",
            f"  Range               : [{self.sweep_values[0]:.4g}, {self.sweep_values[-1]:.4g}]",
            f"  Points              : {len(self.sweep_values)}",
        ]
        if self.param2_name is not None and self.sweep2_values is not None:
            lines.extend([
                f"  Secondary parameter : {self.param2_name}",
                f"  Secondary range     : [{self.sweep2_values[0]:.4g}, {self.sweep2_values[-1]:.4g}]",
                f"  Secondary points    : {len(self.sweep2_values)}",
            ])
        lines.append("  Measured Nodes      : " + ", ".join(k for k in self.node_voltages.keys() if k != "0"))
        lines.append("  Measured Currents   : " + ", ".join(self.branch_currents.keys()))
        lines.append("=" * 60)
        return "\n".join(lines)


def _set_element_param(circuit: Circuit, param_name: str, value: float) -> None:
    """Update parameter value on the designated circuit element."""
    found = False
    p_lower = param_name.strip().lower()

    for el in circuit.elements:
        if el.name.lower() == p_lower:
            found = True
            if isinstance(el, VoltageSource):
                el.voltage = float(value)
            elif isinstance(el, CurrentSource):
                el.current = float(value)
            elif isinstance(el, Resistor):
                if value <= 0:
                    raise ValueError(f"Resistor '{el.name}' resistance must be > 0 (got {value}).")
                el.resistance = float(value)
            elif isinstance(el, ZenerDiode):
                el.Vz = float(value)
            elif isinstance(el, BJT):
                el.beta = float(value)
            elif isinstance(el, MOSFET):
                el.Vth = float(value)
            elif hasattr(el, "value"):
                setattr(el, "value", float(value))
            else:
                raise ValueError(f"Component {el.name!r} has no supported sweep parameter.")
            break

    if not found:
        raise KeyError(
            f"Component '{param_name}' was not found in circuit. "
            f"Available elements: {[el.name for el in circuit.elements]}"
        )


def dc_sweep(
    circuit: Circuit,
    param_name: str,
    start: float,
    stop: float,
    step: float,
    *,
    param2_name: str | None = None,
    start2: float | None = None,
    stop2: float | None = None,
    step2: float | None = None,
) -> DCSweepResult:
    """Perform a DC parametric sweep analysis (.dc sweep).

    Sweeps a component value (voltage source, current source, or resistor)
    across a defined range with optional 2D secondary nested sweep.

    Parameters
    ----------
    circuit : Circuit
        Pre-validated circuit container.
    param_name : str
        Designator of the primary element to sweep (e.g. 'V1', 'R1').
    start : float
        Starting value.
    stop : float
        Ending value.
    step : float
        Step increment.
    param2_name : str | None, optional
        Designator of secondary element to sweep for nested curve families.
    start2 : float | None, optional
        Starting value of secondary parameter.
    stop2 : float | None, optional
        Ending value of secondary parameter.
    step2 : float | None, optional
        Step increment of secondary parameter.

    Returns
    -------
    DCSweepResult
        Container with multidimensional arrays of node voltages and branch currents.

    Examples
    --------
    >>> from circuit_solver import Circuit, VoltageSource, Resistor, dc_sweep
    >>> c = Circuit([VoltageSource("V1", "1", "0", 1.0), Resistor("R1", "1", "0", 10.0)])
    >>> res = dc_sweep(c, "V1", 0.0, 5.0, 1.0)
    >>> res.V("1")
    array([0., 1., 2., 3., 4., 5.])
    """
    def grid(first, last, increment):
        if not np.all(np.isfinite([first, last, increment])) or increment == 0:
            raise ValueError("Sweep bounds and non-zero step must be finite.")
        span = (last - first) / increment
        if span < 0:
            raise ValueError("Sweep step must point from start toward stop.")
        count = int(np.floor(span + 1e-12 * max(1.0, abs(span)))) + 1
        return first + increment * np.arange(count, dtype=float)

    sweep_values = grid(start, stop, step)
    secondary_args = (param2_name, start2, stop2, step2)
    has_secondary = any(value is not None for value in secondary_args)
    if has_secondary and not all(value is not None for value in secondary_args):
        raise ValueError("Secondary sweep requires name, start, stop and step.")
    if has_secondary and param_name.strip().lower() == param2_name.strip().lower():
        raise ValueError("Nested sweeps must use distinct components.")
    sweep2_values = grid(start2, stop2, step2) if has_secondary else None
    # Isolate every mutable element, including when a solver raises mid-sweep.
    circuit = deepcopy(circuit)

    # Track all nodes
    all_nodes = [k for k in circuit.node_map.keys()]
    node_voltages: dict[str, list] = {node: [] for node in all_nodes}
    branch_currents: dict[str, list] = {}

    def record_solution(sol) -> None:
        for node in all_nodes:
            node_voltages[node].append(sol.V(node))
        # Record currents for interesting elements
        for el in circuit.elements:
            if isinstance(el, VoltageSource):
                branch_currents.setdefault(el.name, []).append(sol.I_vsource(el.name))
            elif isinstance(el, Inductor):
                branch_currents.setdefault(el.name, []).append(sol.I_inductor(el.name))
            elif isinstance(el, Transformer):
                branch_currents.setdefault(f"{el.name}_p", []).append(sol.I_transformer_primary(el.name))
                branch_currents.setdefault(f"{el.name}_s", []).append(sol.I_transformer_secondary(el.name))
            elif isinstance(el, Resistor):
                branch_currents.setdefault(el.name, []).append(sol.I_resistor(el.name))
            elif isinstance(el, CurrentSource):
                branch_currents.setdefault(el.name, []).append(sol.I_isource(el.name))
            elif isinstance(el, (Diode, ZenerDiode)):
                branch_currents.setdefault(el.name, []).append(sol.I_diode(el.name))
            elif isinstance(el, BJT):
                cur = sol.I_bjt(el.name)
                branch_currents.setdefault(f"{el.name}_C", []).append(cur["C"])
                branch_currents.setdefault(f"{el.name}_B", []).append(cur["B"])
                branch_currents.setdefault(f"{el.name}_E", []).append(cur["E"])
            elif isinstance(el, MOSFET):
                cur = sol.I_mosfet(el.name)
                branch_currents.setdefault(f"{el.name}_D", []).append(cur["D"])

    if has_secondary and sweep2_values is not None:
        for v2 in sweep2_values:
            _set_element_param(circuit, param2_name, float(v2))
            for v1 in sweep_values:
                _set_element_param(circuit, param_name, float(v1))
                sol = solve(circuit)
                record_solution(sol)

        # Reshape to 2D (num2, num1)
        shape_2d = (len(sweep2_values), len(sweep_values))
        voltages_arr = {
            node: np.array(vals, dtype=float).reshape(shape_2d)
            for node, vals in node_voltages.items()
        }
        currents_arr = {
            name: np.array(vals, dtype=float).reshape(shape_2d)
            for name, vals in branch_currents.items()
        }
        return DCSweepResult(
            param_name=param_name,
            sweep_values=sweep_values,
            node_voltages=voltages_arr,
            branch_currents=currents_arr,
            param2_name=param2_name,
            sweep2_values=sweep2_values,
        )
    else:
        for v1 in sweep_values:
            _set_element_param(circuit, param_name, float(v1))
            sol = solve(circuit)
            record_solution(sol)

        voltages_arr = {
            node: np.array(vals, dtype=float)
            for node, vals in node_voltages.items()
        }
        currents_arr = {
            name: np.array(vals, dtype=float)
            for name, vals in branch_currents.items()
        }
        return DCSweepResult(
            param_name=param_name,
            sweep_values=sweep_values,
            node_voltages=voltages_arr,
            branch_currents=currents_arr,
        )
