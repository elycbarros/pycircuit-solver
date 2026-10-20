"""
validator.py — Pre-solver topological validation and engineering-unit parser.

Two responsibilities
--------------------
1. **Engineering suffix converter** (:func:`parse_value`):
   Translates strings like ``"4.7k"``, ``"10mA"``, ``"220u"``, ``"4k7"``
   into float SI values.

2. **Topological validator** (:class:`CircuitValidator`):
   Checks a list of circuit elements *before* the MNA engine runs:

   * Ground node ``"0"`` is present at least once.
   * No floating (dangling) nodes — every node is connected to ≥ 2 elements.
   * No conflicting voltage sources (two V/E sources directly in parallel).
   * No voltage source directly short-circuiting itself (node_p == node_n).
   * Resistance > 0 (enforced by :class:`~circuit_solver.core.elements.Resistor`
     __post_init__, repeated here for clear error messages).

Raises
------
:class:`CircuitValidationError`
    Raised for any detected topological or physical constraint violation.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
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

__all__ = [
    "parse_value",
    "CircuitValidationError",
    "CircuitValidator",
]

# ---------------------------------------------------------------------------
# Engineering suffix table
# ---------------------------------------------------------------------------

_SUFFIX_MAP: dict[str, float] = {
    "T": 1e12,
    "G": 1e9,
    "MEG": 1e6,
    "K": 1e3,
    "k": 1e3,
    "M": 1e-3,   # SPICE convention: 'M' = milli, 'MEG' = mega
    "m": 1e-3,
    "U": 1e-6,
    "u": 1e-6,
    "N": 1e-9,
    "n": 1e-9,
    "P": 1e-12,
    "p": 1e-12,
    "F": 1e-15,
    "f": 1e-15,
}

# Matches "4k7", "4.7k", "10m", "220uF", "1MEG", etc.
_EMBEDDED_PATTERN = re.compile(
    r"^([+-]?\d*)([TGkKmMuUnNpPfF]|MEG)(\d*)$",
    re.IGNORECASE,
)
_TRAILING_PATTERN = re.compile(
    r"^([+-]?[\d.]+)\s*([TGkKmMuUnNpPfF]|MEG)?$",
    re.IGNORECASE,
)


def parse_value(raw: str | float | int) -> float:
    """Convert an engineering-notation string to a float (SI base units).

    Supports
    --------
    * Plain numbers: ``"100"``, ``3.14``, ``-0.5``
    * Trailing suffix: ``"4.7k"``, ``"10m"``, ``"220u"``
    * Embedded suffix (SpiceSchematic style): ``"4k7"`` → 4700
    * SPICE mega: ``"1MEG"`` → 1,000,000
    * Unit suffix appended (ignored): ``"10mA"`` → 0.01 (A ignored)

    Parameters
    ----------
    raw : str | float | int
        Raw value from netlist or user code.

    Returns
    -------
    float
        Value in SI base units.

    Raises
    ------
    ValueError
        If the string cannot be parsed.

    Examples
    --------
    >>> parse_value("4.7k")
    4700.0
    >>> parse_value("4k7")
    4700.0
    >>> parse_value("220u")
    2.2e-4
    >>> parse_value("1MEG")
    1000000.0
    >>> parse_value(10)
    10.0
    """
    if isinstance(raw, (int, float)):
        return float(raw)

    s = str(raw).strip()
    if not s:
        raise ValueError("Cannot parse empty value.")

    # 1. Embedded suffix check: e.g. "4k7", "4R7", "1M2", "2u2"
    m_emb = re.match(r"^([+-]?\d+)([rR]|[tTgGkKmMuUnNpPfF]|MEG|meg)(\d+)$", s)
    if m_emb:
        pre, suff, post = m_emb.groups()
        suff_upper = suff.upper()
        if suff_upper == "R":
            mult = 1.0
        elif suff_upper == "MEG":
            mult = 1e6
        else:
            mult = _SUFFIX_MAP.get(suff_upper, 1.0)
        return float(f"{pre}.{post}") * mult

    # 2. Standard number + optional engineering prefix + optional unit (e.g. "10mA", "4.7k", "220uF")
    pattern = re.compile(
        r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(MEG|[tTgGkKmMuUnNpPfF])?.*$",
        re.IGNORECASE,
    )
    m = pattern.match(s)
    if m:
        num_str, suff = m.groups()
        try:
            val = float(num_str)
            if suff:
                suff_upper = suff.upper()
                mult = 1e6 if suff_upper == "MEG" else _SUFFIX_MAP.get(suff_upper, 1.0)
                val *= mult
            return val
        except ValueError:
            pass

    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Cannot parse engineering value: '{raw}'")


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------

class CircuitValidationError(Exception):
    """Raised when the circuit topology or element parameters are invalid."""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class CircuitValidator:
    """Pre-solver topological and physical sanity checks.

    Usage
    -----
    >>> validator = CircuitValidator(elements)
    >>> validator.validate()  # raises CircuitValidationError on failure
    """

    def __init__(
        self,
        elements: Sequence[_Element],
        *,
        allow_floating: bool = False,
        allowed_open_nodes: Sequence[str] | None = None,
    ) -> None:
        self._elements = list(elements)
        self.allow_floating = allow_floating
        self.allowed_open_nodes = set(allowed_open_nodes or ())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Run all checks in sequence.  Raises :class:`CircuitValidationError`
        on the *first* detected problem.
        """
        self._check_ground()
        self._check_self_shorts()
        self._check_resistances()
        self._check_controlling_sources()
        self._check_floating_nodes()
        self._check_ground_connectivity()
        self._check_parallel_voltage_sources()
        self._check_voltage_source_loops()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _all_nodes(self) -> list[str]:
        """Return every node label referenced by the element list."""
        nodes: list[str] = []
        for el in self._elements:
            nodes.extend(self._nodes_of(el))
        return nodes

    @staticmethod
    def _nodes_of(el: _Element) -> list[str]:
        """Extract node labels from any element type."""
        if isinstance(el, (Resistor, VoltageSource, CurrentSource, Capacitor, Inductor, CCCS, CCVS)):
            return [el.node_p, el.node_n]
        if isinstance(el, (VCVS, VCCS)):
            return [el.node_p, el.node_n, el.ctrl_p, el.ctrl_n]
        if isinstance(el, OpAmp):
            return [el.in_p, el.in_n, el.out, el.gnd]
        return []

    def _check_ground(self) -> None:
        if "0" not in self._all_nodes():
            raise CircuitValidationError(
                "Ground node '0' not found. Every circuit must have at least "
                "one element connected to node '0' (reference ground)."
            )

    def _check_self_shorts(self) -> None:
        for el in self._elements:
            if isinstance(el, (Resistor, VoltageSource, CurrentSource, Capacitor, Inductor, CCCS, CCVS)):
                if el.node_p == el.node_n:
                    raise CircuitValidationError(
                        f"Element '{el.name}' has node_p == node_n == '{el.node_p}'. "
                        "Both terminals are the same node — this is a short-circuit or "
                        "no-op element."
                    )

    def _check_resistances(self) -> None:
        for el in self._elements:
            if isinstance(el, Resistor) and el.resistance <= 0:
                raise CircuitValidationError(
                    f"Resistor '{el.name}' has non-positive resistance "
                    f"({el.resistance} Ω). Physical resistors must have R > 0."
                )

    def _check_controlling_sources(self) -> None:
        """Verify that current-controlled sources reference a valid VoltageSource."""
        vsource_names = {el.name for el in self._elements if isinstance(el, VoltageSource)}
        for el in self._elements:
            if isinstance(el, (CCCS, CCVS)):
                if el.vctrl_name not in vsource_names:
                    raise CircuitValidationError(
                        f"Element '{el.name}' references controlling voltage source '{el.vctrl_name}', "
                        "which does not exist in the circuit. In MNA/SPICE, current-controlled sources "
                        "must reference a defined independent VoltageSource (or a 0V dummy source)."
                    )

    def _check_floating_nodes(self) -> None:
        """Every non-ground node must appear in ≥ 2 element connections."""
        if self.allow_floating:
            return
        count: dict[str, int] = defaultdict(int)
        for node in self._all_nodes():
            count[node] += 1
        floating = [
            n for n, c in count.items()
            if c < 2 and n != "0" and n not in self.allowed_open_nodes
        ]
        if floating:
            raise CircuitValidationError(
                f"Floating (dangling) node(s) detected: {sorted(floating)}. "
                "Each non-ground node must be connected to at least 2 element terminals."
            )

    def _check_parallel_voltage_sources(self) -> None:
        """Detect two ideal voltage sources directly in parallel (same node pair)."""
        v_sources: list[tuple[str, str, str]] = []  # (name, node_p, node_n)
        for el in self._elements:
            if isinstance(el, (VoltageSource, VCVS, CCVS)):
                v_sources.append((el.name, el.node_p, el.node_n))
        seen: dict[frozenset[str], str] = {}
        for name, np, nn in v_sources:
            key = frozenset({np, nn})
            if key in seen:
                raise CircuitValidationError(
                    f"Conflicting voltage sources: '{seen[key]}' and '{name}' are "
                    f"connected between the same nodes ('{np}', '{nn}'). Parallel "
                    "ideal voltage sources with different values are physically impossible."
                )
            seen[key] = name

    def _check_ground_connectivity(self) -> None:
        """Every node must have a conductive path to reference ground '0'."""
        if self.allow_floating:
            return

        adj: dict[str, set[str]] = defaultdict(set)
        for el in self._elements:
            if isinstance(el, (Resistor, VoltageSource, CurrentSource, Capacitor, Inductor, CCCS, CCVS)):
                adj[el.node_p].add(el.node_n)
                adj[el.node_n].add(el.node_p)
            elif isinstance(el, (VCVS, VCCS)):
                adj[el.node_p].add(el.node_n)
                adj[el.node_n].add(el.node_p)
            elif isinstance(el, OpAmp):
                adj[el.out].add(el.gnd)
                adj[el.gnd].add(el.out)

        visited: set[str] = set()
        queue = deque(["0"])
        visited.add("0")
        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        all_nodes = set(self._all_nodes())
        unreachable = [
            n for n in all_nodes
            if n not in visited and n not in self.allowed_open_nodes
        ]
        if unreachable:
            raise CircuitValidationError(
                f"Disconnected subcircuit detected: node(s) {sorted(unreachable)} have no "
                "conductive path to reference ground '0'. Every subcircuit must be connected to ground."
            )

    def _check_voltage_source_loops(self) -> None:
        """Detect closed loops formed exclusively of ideal voltage sources."""
        v_sources: list[tuple[str, str, str]] = []
        for el in self._elements:
            if isinstance(el, (VoltageSource, VCVS, CCVS)):
                v_sources.append((el.name, el.node_p, el.node_n))

        adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for name, np, nn in v_sources:
            adj[np].append((nn, name))
            adj[nn].append((np, name))

        visited: set[str] = set()

        def dfs(node: str, edge_used: str | None, path: list[str]) -> list[str] | None:
            visited.add(node)
            for neighbor, s_name in adj[node]:
                if s_name == edge_used:
                    continue
                if neighbor in path:
                    idx_cycle = path.index(neighbor)
                    return path[idx_cycle:] + [neighbor]
                if neighbor not in visited:
                    res = dfs(neighbor, s_name, path + [neighbor])
                    if res:
                        return res
            return None

        for n in list(adj.keys()):
            if n not in visited:
                cycle = dfs(n, None, [n])
                if cycle:
                    cycle_nodes = sorted(set(cycle))
                    raise CircuitValidationError(
                        f"Closed loop of ideal voltage sources detected involving nodes {cycle_nodes}. "
                        "A closed loop of ideal voltage sources is physically impossible (KVL violation)."
                    )
