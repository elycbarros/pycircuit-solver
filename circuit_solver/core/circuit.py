"""
circuit.py — Circuit container and node-index mapping.

:class:`Circuit` is the central data structure that collects elements,
assigns integer indices to node labels (excluding ground '0'), and
exposes the ordered lists consumed by the MNA engine.

Node ordering
-------------
Ground node ``'0'`` is **never** assigned an index (it is the implicit
reference, always at 0 V).  All other nodes are assigned consecutive
indices 0, 1, 2, ... in the order they are first encountered.

Auxiliary variable ordering
---------------------------
After the node block, MNA appends one auxiliary current variable per
voltage-enforcing element (VoltageSource, VCVS, OpAmp output branch).
These are tracked via :attr:`auxiliary_map`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    OpAmp,
    Capacitor,
    Inductor,
    MutualCoupling,
    Transformer,
    CCCS,
    CCVS,
    Diode,
    ZenerDiode,
    BJT,
    MOSFET,
    _Element,
)
from circuit_solver.core.validator import CircuitValidator
from circuit_solver.core.subcircuit import HierarchyTree

__all__ = ["Circuit"]


class Circuit:
    """Container for circuit topology with automatic node indexing.

    Parameters
    ----------
    elements : list of _Element
        Circuit element instances.  Engineering-suffix strings in element
        values are **not** parsed here — callers must use :func:`parse_value`
        before constructing elements, or use :meth:`from_elements`.
    validate : bool
        If ``True`` (default) run :class:`CircuitValidator` before returning.
    hierarchy : HierarchyTree | None, optional
        Hierarchical subcircuit tree metadata if expanded from SPICE netlist.

    Attributes
    ----------
    elements : list[_Element]
        Ordered list of all circuit elements.
    node_map : dict[str, int]
        Mapping from node label to 0-based integer index.
        Ground ``'0'`` is excluded.
    n_nodes : int
        Number of non-ground nodes (= size of the nodal voltage sub-block).
    auxiliary_map : dict[str, int]
        Mapping from element name to index in the auxiliary-variable block.
    n_aux : int
        Number of auxiliary current variables.
    size : int
        Total size of the MNA system: ``n_nodes + n_aux``.
    hierarchy : HierarchyTree | None
        Hierarchy tree containing subcircuit definitions, instances, and
        pedagogical white-box trace.
    """

    def __init__(
        self,
        elements: Sequence[_Element],
        validate: bool = True,
        *,
        allow_floating: bool = False,
        allowed_open_nodes: Sequence[str] | None = None,
        hierarchy: HierarchyTree | None = None,
    ) -> None:
        self.elements: list[_Element] = list(elements)
        self.hierarchy: HierarchyTree | None = hierarchy

        if validate:
            CircuitValidator(
                self.elements,
                allow_floating=allow_floating,
                allowed_open_nodes=allowed_open_nodes,
            ).validate()

        self.node_map: dict[str, int] = {}
        self.auxiliary_map: dict[str, int] = {}
        self._build_maps()

    @classmethod
    def from_netlist(
        cls,
        path_or_text: str | Path,
        validate: bool = True,
        *,
        allow_floating: bool = False,
        allowed_open_nodes: Sequence[str] | None = None,
    ) -> "Circuit":
        """Construct a Circuit directly from a SPICE netlist file or string.

        Automatically expands all .SUBCKT definitions and 'X' instances while
        preserving hierarchical pedagogical metadata.
        """
        from circuit_solver.io.parser import parse_netlist_with_hierarchy

        elements, hierarchy = parse_netlist_with_hierarchy(path_or_text)
        return cls(
            elements,
            validate=validate,
            allow_floating=allow_floating,
            allowed_open_nodes=allowed_open_nodes,
            hierarchy=hierarchy,
        )

    def explain_hierarchy(self) -> str:
        """Pedagogical white-box explanation of subcircuit expansion."""
        if self.hierarchy is not None:
            return self.hierarchy.explain(self)
        return "Circuito Plano (Sem subcircuitos hierárquicos)."

    def hierarchy_tree(self) -> str:
        """ASCII tree visualization of circuit hierarchy."""
        if self.hierarchy is not None:
            return self.hierarchy.tree_view()
        return "Circuito Plano (Sem subcircuitos hierárquicos)."

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register_node(self, label: str) -> None:
        if label != "0" and label not in self.node_map:
            self.node_map[label] = len(self.node_map)

    def _build_maps(self) -> None:
        """Assign node indices and auxiliary variable indices."""
        # Pass 1: register nodes in element order
        for el in self.elements:
            for node in self._nodes_of(el):
                self._register_node(node)

        # Pass 2: auxiliary variables for voltage-enforcing elements
        aux_idx = 0
        for el in self.elements:
            if isinstance(el, (VoltageSource, VCVS, CCVS, OpAmp)):
                self.auxiliary_map[el.name] = aux_idx
                aux_idx += 1

        self.n_nodes: int = len(self.node_map)
        self.n_aux: int = len(self.auxiliary_map)
        self.size: int = self.n_nodes + self.n_aux

    @property
    def has_nonlinear(self) -> bool:
        """True if the circuit contains any non-linear semiconductor elements."""
        return any(isinstance(el, (Diode, ZenerDiode, BJT, MOSFET)) for el in self.elements)

    @staticmethod
    def _nodes_of(el: _Element) -> list[str]:
        if isinstance(el, (Resistor, VoltageSource, CurrentSource, Capacitor, Inductor, CCCS, CCVS, Diode, ZenerDiode)):
            return [el.node_p, el.node_n]
        if isinstance(el, (VCVS, VCCS)):
            return [el.node_p, el.node_n, el.ctrl_p, el.ctrl_n]
        if isinstance(el, OpAmp):
            return [el.in_p, el.in_n, el.out, el.gnd]
        if isinstance(el, BJT):
            return [el.node_c, el.node_b, el.node_e]
        if isinstance(el, MOSFET):
            return [el.node_d, el.node_g, el.node_s]
        if isinstance(el, Transformer):
            return [el.p1, el.n1, el.p2, el.n2]
        if isinstance(el, MutualCoupling):
            return []
        return []

    def get_coupled_inductor_names(self) -> set[str]:
        """Names of individual Inductor elements that participate in a MutualCoupling."""
        names = set()
        for el in self.elements:
            if isinstance(el, MutualCoupling):
                names.add(el.ind1_name)
                names.add(el.ind2_name)
        return names

    def get_coupled_pairs(self) -> list[dict]:
        """Resolve all magnetically coupled pairs (both MutualCoupling and Transformer).

        Returns a list of dictionaries with resolved parameters:
            - name: identifier
            - p1, n1: primary terminal nodes
            - p2, n2: secondary terminal nodes
            - L1, L2: self inductances in H
            - coupling: coupling coefficient k
            - M: mutual inductance in H
            - Gamma: 2x2 inverse inductance matrix (H^-1)
            - ind1_name: inductor 1 name (or transformer primary name)
            - ind2_name: inductor 2 name (or transformer secondary name)
            - is_transformer: bool
        """
        pairs: list[dict] = []
        ind_by_name = {el.name: el for el in self.elements if isinstance(el, Inductor)}

        # 1. MutualCoupling (SPICE K element)
        for el in self.elements:
            if isinstance(el, MutualCoupling):
                if el.ind1_name not in ind_by_name or el.ind2_name not in ind_by_name:
                    continue
                ind1 = ind_by_name[el.ind1_name]
                ind2 = ind_by_name[el.ind2_name]
                L1, L2 = ind1.inductance, ind2.inductance
                k = el.coupling
                # Clamp k to avoid division by zero if |k| == 1
                k_clamped = min(max(k, -0.9999999), 0.9999999)
                M = k * np.sqrt(L1 * L2)
                M_clamped = k_clamped * np.sqrt(L1 * L2)
                det = L1 * L2 - M_clamped**2
                Gamma = np.array([
                    [L2 / det, -M_clamped / det],
                    [-M_clamped / det, L1 / det]
                ], dtype=float)

                pairs.append({
                    "name": el.name,
                    "p1": ind1.node_p,
                    "n1": ind1.node_n,
                    "p2": ind2.node_p,
                    "n2": ind2.node_n,
                    "L1": L1,
                    "L2": L2,
                    "coupling": k,
                    "M": M,
                    "Gamma": Gamma,
                    "ind1_name": ind1.name,
                    "ind2_name": ind2.name,
                    "is_transformer": False,
                })

        # 2. Transformer
        for el in self.elements:
            if isinstance(el, Transformer):
                L1, L2 = el.L1, el.L2
                k = el.coupling
                k_clamped = min(max(k, -0.9999999), 0.9999999)
                M = k * np.sqrt(L1 * L2)
                M_clamped = k_clamped * np.sqrt(L1 * L2)
                det = L1 * L2 - M_clamped**2
                Gamma = np.array([
                    [L2 / det, -M_clamped / det],
                    [-M_clamped / det, L1 / det]
                ], dtype=float)

                pairs.append({
                    "name": el.name,
                    "p1": el.p1,
                    "n1": el.n1,
                    "p2": el.p2,
                    "n2": el.n2,
                    "L1": L1,
                    "L2": L2,
                    "coupling": k,
                    "M": M,
                    "Gamma": Gamma,
                    "ind1_name": f"{el.name}_p",
                    "ind2_name": f"{el.name}_s",
                    "is_transformer": True,
                })

        return pairs

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def node_index(self, label: str) -> int | None:
        """Return the MNA row/column index for ``label``, or ``None`` for ground."""
        if str(label).strip() == "0":
            return None
        lbl = str(label).strip()
        if lbl in self.node_map:
            return self.node_map[lbl]
        for k, idx in self.node_map.items():
            if k.lower() == lbl.lower():
                return idx
        return None

    def aux_index(self, name: str) -> int:
        """Return the auxiliary-block index for element ``name``."""
        return self.n_nodes + self.auxiliary_map[name]

    @property
    def nodes(self) -> list[str]:
        """Sorted list of non-ground node labels (by assigned index)."""
        return sorted(self.node_map, key=self.node_map.__getitem__)

    def __repr__(self) -> str:
        return (
            f"Circuit(n_nodes={self.n_nodes}, n_aux={self.n_aux}, "
            f"elements={len(self.elements)})"
        )
