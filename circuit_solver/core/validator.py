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
from collections import defaultdict
from typing import Sequence

from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    OpAmp,
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

    # --- Try embedded suffix first: "4k7" ---
    m = _EMBEDDED_PATTERN.match(s)
    if m:
        pre, suffix, post = m.group(1), m.group(2), m.group(3)
        suffix_key = suffix.upper() if suffix.upper() in ("MEG",) else suffix
        multiplier = _SUFFIX_MAP.get(suffix_key, _SUFFIX_MAP.get(suffix.upper(), 1.0))
        left = float(pre) if pre else 0.0
        right = float(f"0.{post}") if post else 0.0
        return (left + right) * multiplier

    # --- Strip trailing non-numeric unit character(s) like "A" in "10mA" ---
    # Remove a final alpha char that is NOT a known suffix (e.g. 'A', 'V', 'H')
    cleaned = re.sub(r"[A-Za-z]+$", lambda mo: mo.group() if mo.group().upper() in _SUFFIX_MAP or mo.group().upper() == "MEG" else "", s)
    # Re-attach suffix if it was eaten
    m2 = _TRAILING_PATTERN.match(cleaned if cleaned else s)
    if not m2:
        # Last attempt: strip all trailing alpha
        s_stripped = re.sub(r"[A-Za-z]+$", "", s)
        try:
            return float(s_stripped)
        except ValueError:
            raise ValueError(f"Cannot parse engineering value: '{raw}'")

    num_part = m2.group(1)
    suffix = m2.group(2)
    value = float(num_part)
    if suffix:
        suffix_key = suffix.upper() if suffix.upper() in ("MEG",) else suffix
        multiplier = _SUFFIX_MAP.get(suffix_key, _SUFFIX_MAP.get(suffix.upper(), 1.0))
        value *= multiplier
    return value


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

    def __init__(self, elements: Sequence[_Element]) -> None:
        self._elements = list(elements)

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
        self._check_floating_nodes()
        self._check_parallel_voltage_sources()

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
        if isinstance(el, (Resistor, VoltageSource, CurrentSource)):
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
            if isinstance(el, (Resistor, VoltageSource, CurrentSource)):
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

    def _check_floating_nodes(self) -> None:
        """Every non-ground node must appear in ≥ 2 element connections."""
        count: dict[str, int] = defaultdict(int)
        for node in self._all_nodes():
            count[node] += 1
        floating = [n for n, c in count.items() if c < 2 and n != "0"]
        if floating:
            raise CircuitValidationError(
                f"Floating (dangling) node(s) detected: {sorted(floating)}. "
                "Each non-ground node must be connected to at least 2 element terminals."
            )

    def _check_parallel_voltage_sources(self) -> None:
        """Detect two ideal voltage sources directly in parallel (same node pair)."""
        v_sources: list[tuple[str, str, str]] = []  # (name, node_p, node_n)
        for el in self._elements:
            if isinstance(el, (VoltageSource, VCVS)):
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
