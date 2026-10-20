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

from typing import Sequence

from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    OpAmp,
    Capacitor,
    Inductor,
    CCCS,
    CCVS,
    _Element,
)
from circuit_solver.core.validator import CircuitValidator, parse_value

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
    """

    def __init__(
        self,
        elements: Sequence[_Element],
        validate: bool = True,
        *,
        allow_floating: bool = False,
        allowed_open_nodes: Sequence[str] | None = None,
    ) -> None:
        self.elements: list[_Element] = list(elements)

        if validate:
            CircuitValidator(
                self.elements,
                allow_floating=allow_floating,
                allowed_open_nodes=allowed_open_nodes,
            ).validate()

        self.node_map: dict[str, int] = {}
        self.auxiliary_map: dict[str, int] = {}
        self._build_maps()

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

    @staticmethod
    def _nodes_of(el: _Element) -> list[str]:
        if isinstance(el, (Resistor, VoltageSource, CurrentSource, Capacitor, Inductor, CCCS, CCVS)):
            return [el.node_p, el.node_n]
        if isinstance(el, (VCVS, VCCS)):
            return [el.node_p, el.node_n, el.ctrl_p, el.ctrl_n]
        if isinstance(el, OpAmp):
            return [el.in_p, el.in_n, el.out, el.gnd]
        return []

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
