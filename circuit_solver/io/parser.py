"""
parser.py — SPICE-subset netlist reader (.cir / .net / .sp files).

Supported element lines
-----------------------
::

    Rname  n+  n-  value          Resistor
    Vname  n+  n-  DC value       Independent voltage source
    Iname  n+  n-  DC value       Independent current source
    Ename  n+  n-  nc+ nc-  gain  VCVS
    Gname  n+  n-  nc+ nc-  gm    VCCS

Op-Amp (Antigravity extension, not standard SPICE):
::

    Aname  in+  in-  out  [gnd]   Ideal Op-Amp

Netlist conventions
-------------------
* Case-insensitive element designators and node labels.
* First line is the title and is silently ignored.
* Lines starting with ``*`` are comments.
* ``.end`` terminates parsing.
* ``DC`` keyword before the value in V/I lines is optional.
* Values may use engineering suffixes: ``k``, ``m``, ``u``, ``n``, ``p``,
  ``MEG`` (see :func:`~circuit_solver.core.validator.parse_value`).

Example netlist
---------------
::

    Voltage divider example
    V1  1  0  DC  10
    R1  1  2  4k7
    R2  2  0  2.2k
    .end

Usage
-----
>>> from circuit_solver.io.parser import parse_netlist
>>> elements = parse_netlist("myfile.cir")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

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
    _Element,
)
from circuit_solver.core.validator import parse_value

__all__ = ["parse_netlist", "parse_netlist_string"]


def _strip_comment(line: str) -> str:
    """Remove inline SPICE comments (everything after ';' or '$')."""
    for ch in (";", "$"):
        idx = line.find(ch)
        if idx != -1:
            line = line[:idx]
    return line.strip()


def _parse_line(line: str, lineno: int) -> _Element | None:
    """Parse a single netlist element line.  Returns ``None`` for empty/directive lines."""
    line = _strip_comment(line)
    if not line:
        return None

    tokens = line.split()
    designator = tokens[0].upper()
    prefix = designator[0]

    try:
        if prefix == "R":
            # R<name>  n+  n-  value
            name, np, nn, val_str = tokens[0], tokens[1], tokens[2], tokens[3]
            return Resistor(name.upper(), np.lower(), nn.lower(), parse_value(val_str))

        elif prefix == "C":
            # C<name>  n+  n-  value
            name, np, nn, val_str = tokens[0], tokens[1], tokens[2], tokens[3]
            return Capacitor(name.upper(), np.lower(), nn.lower(), parse_value(val_str))

        elif prefix == "L":
            # L<name>  n+  n-  value
            name, np, nn, val_str = tokens[0], tokens[1], tokens[2], tokens[3]
            return Inductor(name.upper(), np.lower(), nn.lower(), parse_value(val_str))

        elif prefix == "V":
            # V<name>  n+  n-  [DC val]  [AC mag [phase]]
            name, np, nn = tokens[0], tokens[1], tokens[2]
            dc_val = 0.0
            ac_mag = 0.0
            ac_phase = 0.0
            idx_t = 3
            while idx_t < len(tokens):
                tok_u = tokens[idx_t].upper()
                if tok_u == "DC":
                    idx_t += 1
                    dc_val = parse_value(tokens[idx_t])
                elif tok_u == "AC":
                    idx_t += 1
                    ac_mag = parse_value(tokens[idx_t])
                    if idx_t + 1 < len(tokens) and not tokens[idx_t + 1].isalpha():
                        idx_t += 1
                        ac_phase = parse_value(tokens[idx_t])
                else:
                    # Positional value
                    dc_val = parse_value(tokens[idx_t])
                idx_t += 1
            return VoltageSource(name.upper(), np.lower(), nn.lower(), dc_val, ac_mag=ac_mag, ac_phase=ac_phase)

        elif prefix == "I":
            # I<name>  n+  n-  [DC val]  [AC mag [phase]]
            name, np, nn = tokens[0], tokens[1], tokens[2]
            dc_val = 0.0
            ac_mag = 0.0
            ac_phase = 0.0
            idx_t = 3
            while idx_t < len(tokens):
                tok_u = tokens[idx_t].upper()
                if tok_u == "DC":
                    idx_t += 1
                    dc_val = parse_value(tokens[idx_t])
                elif tok_u == "AC":
                    idx_t += 1
                    ac_mag = parse_value(tokens[idx_t])
                    if idx_t + 1 < len(tokens) and not tokens[idx_t + 1].isalpha():
                        idx_t += 1
                        ac_phase = parse_value(tokens[idx_t])
                else:
                    dc_val = parse_value(tokens[idx_t])
                idx_t += 1
            return CurrentSource(name.upper(), np.lower(), nn.lower(), dc_val, ac_mag=ac_mag, ac_phase=ac_phase)

        elif prefix == "E":
            # E<name>  n+  n-  nc+  nc-  gain
            name, np, nn, cp, cn, gain_str = (
                tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5]
            )
            return VCVS(
                name.upper(), np.lower(), nn.lower(),
                cp.lower(), cn.lower(), parse_value(gain_str)
            )

        elif prefix == "G":
            # G<name>  n+  n-  nc+  nc-  gm
            name, np, nn, cp, cn, gm_str = (
                tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5]
            )
            return VCCS(
                name.upper(), np.lower(), nn.lower(),
                cp.lower(), cn.lower(), parse_value(gm_str)
            )

        elif prefix == "F":
            # F<name>  n+  n-  Vctrl  gain
            name, np, nn, vctrl, gain_str = (
                tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]
            )
            return CCCS(name.upper(), np.lower(), nn.lower(), vctrl.upper(), parse_value(gain_str))

        elif prefix == "H":
            # H<name>  n+  n-  Vctrl  transresistance
            name, np, nn, vctrl, rm_str = (
                tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]
            )
            return CCVS(name.upper(), np.lower(), nn.lower(), vctrl.upper(), parse_value(rm_str))

        elif prefix in ("A", "U"):
            # A<name>  in+  in-  out  [gnd]
            name, inp, inn, out = tokens[0], tokens[1], tokens[2], tokens[3]
            gnd = tokens[4].lower() if len(tokens) > 4 else "0"
            return OpAmp(name.upper(), inp.lower(), inn.lower(), out.lower(), gnd)

        else:
            # Unknown / unsupported element — warn and skip
            import warnings
            warnings.warn(
                f"Line {lineno}: unsupported element '{tokens[0]}' — skipped.",
                SyntaxWarning,
                stacklevel=4,
            )
            return None

    except (IndexError, ValueError) as exc:
        raise SyntaxError(
            f"Netlist parse error at line {lineno}: '{line}'\n  → {exc}"
        ) from exc


def parse_netlist_string(text: str) -> list[_Element]:
    """Parse a SPICE netlist from a multi-line string.

    Parameters
    ----------
    text : str
        Full netlist content as a string.

    Returns
    -------
    list[_Element]
        Ordered list of circuit elements.
    """
    elements: list[_Element] = []
    lines = text.splitlines()
    # First line is always the title — skip it
    body = lines[1:] if lines else []

    for lineno, raw in enumerate(body, start=2):
        stripped = raw.strip()
        # Skip comments and blank lines
        if not stripped or stripped.startswith("*") or stripped.startswith("+"):
            continue
        # Stop at .end
        if stripped.lower().startswith(".end"):
            break
        # Skip other directives (.tran, .ac, .op, .model, etc.)
        if stripped.startswith("."):
            continue

        el = _parse_line(stripped, lineno)
        if el is not None:
            elements.append(el)

    return elements


def parse_netlist(path: str | Path) -> list[_Element]:
    """Read and parse a SPICE netlist file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.cir`` / ``.net`` / ``.sp`` file.

    Returns
    -------
    list[_Element]
        Ordered list of circuit elements.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    SyntaxError
        If a netlist line cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Netlist file not found: '{path}'")
    return parse_netlist_string(p.read_text(encoding="utf-8"))
