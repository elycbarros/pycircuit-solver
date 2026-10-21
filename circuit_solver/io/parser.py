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
    MutualCoupling,
    Transformer,
    OpAmp,
    Diode,
    ZenerDiode,
    BJT,
    MOSFET,
    PulseWaveform,
    SineWaveform,
    _Element,
)
from circuit_solver.core.validator import parse_value
from circuit_solver.core.subcircuit import HierarchyTree, SubcircuitExpander, tokenize_spice_line

__all__ = ["parse_netlist", "parse_netlist_string", "parse_netlist_with_hierarchy", "parse_dc_directive"]


def _strip_comment(line: str) -> str:
    """Remove inline SPICE comments (everything after ';' or '$')."""
    for ch in (";", "$"):
        idx = line.find(ch)
        if idx != -1:
            line = line[:idx]
    return line.strip()


def _extract_waveform(line: str) -> object | None:
    """Extract SPICE PULSE(...) or SIN(...) waveform from a source line."""
    u = line.upper()
    if "PULSE" in u:
        m = re.search(r"PULSE\s*\(([^)]+)\)", line, re.IGNORECASE)
        if m:
            args = [parse_value(x) for x in m.group(1).replace(",", " ").split()]
            v1 = args[0] if len(args) > 0 else 0.0
            v2 = args[1] if len(args) > 1 else 1.0
            td = args[2] if len(args) > 2 else 0.0
            tr = args[3] if len(args) > 3 else 1e-9
            tf = args[4] if len(args) > 4 else 1e-9
            pw = args[5] if len(args) > 5 else 1.0
            per = args[6] if len(args) > 6 else 2.0
            return PulseWaveform(v1, v2, td, tr, tf, pw, per)
    elif "SIN" in u:
        m = re.search(r"SIN\s*\(([^)]+)\)", line, re.IGNORECASE)
        if m:
            args = [parse_value(x) for x in m.group(1).replace(",", " ").split()]
            vo = args[0] if len(args) > 0 else 0.0
            va = args[1] if len(args) > 1 else 1.0
            freq = args[2] if len(args) > 2 else 60.0
            td = args[3] if len(args) > 3 else 0.0
            theta = args[4] if len(args) > 4 else 0.0
            return SineWaveform(vo, va, freq, td, theta)
    return None


def _parse_line(line: str, lineno: int) -> _Element | None:
    """Parse a single netlist element line.  Returns ``None`` for empty/directive lines."""
    line = _strip_comment(line)
    if not line:
        return None

    tokens = tokenize_spice_line(line)
    designator = tokens[0].upper()
    base_name = designator.split(".")[-1]
    prefix = base_name[0]

    try:
        if prefix == "R":
            # R<name>  n+  n-  value
            name, np, nn, val_str = tokens[0], tokens[1], tokens[2], tokens[3]
            return Resistor(name.upper(), np.lower(), nn.lower(), parse_value(val_str))

        elif prefix == "C":
            # C<name>  n+  n-  value  [IC=...]
            name, np, nn, val_str = tokens[0], tokens[1], tokens[2], tokens[3]
            ic = 0.0
            for t in tokens[4:]:
                if t.upper().startswith("IC="):
                    ic = parse_value(t[3:])
            return Capacitor(name.upper(), np.lower(), nn.lower(), parse_value(val_str), ic=ic)

        elif prefix == "L":
            # L<name>  n+  n-  value  [IC=...]
            name, np, nn, val_str = tokens[0], tokens[1], tokens[2], tokens[3]
            ic = 0.0
            for t in tokens[4:]:
                if t.upper().startswith("IC="):
                    ic = parse_value(t[3:])
            return Inductor(name.upper(), np.lower(), nn.lower(), parse_value(val_str), ic=ic)

        elif prefix == "V":
            # V<name>  n+  n-  [DC val]  [AC mag [phase]]  [PULSE(...)] [SIN(...)]
            name, np, nn = tokens[0], tokens[1], tokens[2]
            dc_val = 0.0
            ac_mag = 0.0
            ac_phase = 0.0
            waveform = _extract_waveform(line)

            idx_t = 3
            while idx_t < len(tokens):
                tok_u = tokens[idx_t].upper()
                if tok_u == "DC":
                    idx_t += 1
                    dc_val = parse_value(tokens[idx_t])
                elif tok_u == "AC":
                    idx_t += 1
                    ac_mag = parse_value(tokens[idx_t])
                    if idx_t + 1 < len(tokens) and not tokens[idx_t + 1].isalpha() and "(" not in tokens[idx_t + 1]:
                        idx_t += 1
                        ac_phase = parse_value(tokens[idx_t])
                elif not tok_u.startswith("PULSE") and not tok_u.startswith("SIN") and "(" not in tok_u and ")" not in tok_u:
                    try:
                        dc_val = parse_value(tokens[idx_t])
                    except Exception:
                        pass
                idx_t += 1

            if waveform is not None and isinstance(waveform, PulseWaveform) and dc_val == 0.0:
                dc_val = waveform.v1

            return VoltageSource(
                name.upper(), np.lower(), nn.lower(), dc_val,
                ac_mag=ac_mag, ac_phase=ac_phase, waveform=waveform
            )

        elif prefix == "I":
            # I<name>  n+  n-  [DC val]  [AC mag [phase]]  [PULSE(...)] [SIN(...)]
            name, np, nn = tokens[0], tokens[1], tokens[2]
            dc_val = 0.0
            ac_mag = 0.0
            ac_phase = 0.0
            waveform = _extract_waveform(line)

            idx_t = 3
            while idx_t < len(tokens):
                tok_u = tokens[idx_t].upper()
                if tok_u == "DC":
                    idx_t += 1
                    dc_val = parse_value(tokens[idx_t])
                elif tok_u == "AC":
                    idx_t += 1
                    ac_mag = parse_value(tokens[idx_t])
                    if idx_t + 1 < len(tokens) and not tokens[idx_t + 1].isalpha() and "(" not in tokens[idx_t + 1]:
                        idx_t += 1
                        ac_phase = parse_value(tokens[idx_t])
                elif not tok_u.startswith("PULSE") and not tok_u.startswith("SIN") and "(" not in tok_u and ")" not in tok_u:
                    try:
                        dc_val = parse_value(tokens[idx_t])
                    except Exception:
                        pass
                idx_t += 1

            return CurrentSource(
                name.upper(), np.lower(), nn.lower(), dc_val,
                ac_mag=ac_mag, ac_phase=ac_phase, waveform=waveform
            )

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

        elif prefix == "D":
            # D<name> n+ n- [model] [IS=val] [N=val] [VT=val]
            name, np, nn = tokens[0], tokens[1], tokens[2]
            is_val = 1e-14
            n_factor = 1.0
            vt = 0.02585
            for extra in tokens[3:]:
                u_tok = extra.upper()
                if u_tok.startswith("IS="):
                    is_val = parse_value(extra.split("=")[1])
                elif u_tok.startswith("N="):
                    n_factor = parse_value(extra.split("=")[1])
                elif u_tok.startswith("VT="):
                    vt = parse_value(extra.split("=")[1])
            return Diode(name.upper(), np.lower(), nn.lower(), Is=is_val, n=n_factor, Vt=vt)

        elif prefix == "Z":
            # Z<name> n+ n- Vz [Rz=val]
            name, np, nn, vz_str = tokens[0], tokens[1], tokens[2], tokens[3]
            vz = parse_value(vz_str)
            rz = 10.0
            for extra in tokens[4:]:
                u_tok = extra.upper()
                if u_tok.startswith("RZ="):
                    rz = parse_value(extra.split("=")[1])
            return ZenerDiode(name.upper(), np.lower(), nn.lower(), Vz=vz, Rz=rz)

        elif prefix == "Q":
            # Q<name> nc nb ne [model] [NPN|PNP] [BETA=val]
            name, nc, nb, ne = tokens[0], tokens[1], tokens[2], tokens[3]
            polarity = "NPN"
            beta = 100.0
            for extra in tokens[4:]:
                u_tok = extra.upper()
                if u_tok in ("NPN", "PNP"):
                    polarity = u_tok
                elif u_tok.startswith("BETA=") or u_tok.startswith("BF="):
                    beta = parse_value(extra.split("=")[1])
            return BJT(name.upper(), nc.lower(), nb.lower(), ne.lower(), polarity=polarity, beta=beta)

        elif prefix == "M":
            # M<name> nd ng ns [model] [NMOS|PMOS] [VTH=val] [KP=val] [LAMBDA=val]
            name, nd, ng, ns = tokens[0], tokens[1], tokens[2], tokens[3]
            polarity = "NMOS"
            vth = 1.0
            kp = 1e-3
            lam = 0.0
            for extra in tokens[4:]:
                u_tok = extra.upper()
                if u_tok in ("NMOS", "PMOS"):
                    polarity = u_tok
                elif u_tok.startswith("VTH=") or u_tok.startswith("VT="):
                    vth = parse_value(extra.split("=")[1])
                elif u_tok.startswith("KP="):
                    kp = parse_value(extra.split("=")[1])
                elif u_tok.startswith("LAMBDA="):
                    lam = parse_value(extra.split("=")[1])
            return MOSFET(name.upper(), nd.lower(), ng.lower(), ns.lower(), polarity=polarity, Vth=vth, Kp=kp, lambda_param=lam)

        elif prefix == "K":
            # K<name>  L1  L2  [coupling]
            name, ind1, ind2 = tokens[0], tokens[1], tokens[2]
            k_val = parse_value(tokens[3]) if len(tokens) > 3 else 0.999
            return MutualCoupling(name.upper(), ind1.upper(), ind2.upper(), coupling=k_val)

        elif prefix == "T":
            # T<name>  p1  n1  p2  n2  L1  L2  [coupling]
            name, p1, n1, p2, n2, l1_str, l2_str = (
                tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6]
            )
            k_val = parse_value(tokens[7]) if len(tokens) > 7 else 0.999
            return Transformer(
                name.upper(), p1.lower(), n1.lower(), p2.lower(), n2.lower(),
                L1=parse_value(l1_str), L2=parse_value(l2_str), coupling=k_val
            )

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


def parse_netlist_with_hierarchy(
    text_or_path: str | Path,
    base_dir: Path | None = None,
) -> tuple[list[_Element], HierarchyTree]:
    """Parse a SPICE netlist string or file, expanding all .SUBCKT blocks into elements
    while constructing a White-Box HierarchyTree for pedagogical introspection.

    Parameters
    ----------
    text_or_path : str or Path
        Netlist content as string, or path to a .cir/.net/.sp file.
    base_dir : Path | None, optional
        Base directory for resolving .include directives.

    Returns
    -------
    tuple[list[_Element], HierarchyTree]
        Flattened elements and the pedagogical hierarchy tree.
    """
    path_candidate = Path(str(text_or_path))
    if path_candidate.exists() and path_candidate.is_file():
        text = path_candidate.read_text(encoding="utf-8")
        if base_dir is None:
            base_dir = path_candidate.parent
    else:
        text = str(text_or_path)

    raw_lines = text.splitlines()
    idx_title = 0
    while idx_title < len(raw_lines) and not raw_lines[idx_title].strip():
        idx_title += 1
    body = raw_lines[idx_title + 1:] if idx_title < len(raw_lines) else []

    # Pre-process: merge continuation lines starting with '+'
    logical_lines: list[tuple[int, str]] = []
    for lineno, raw in enumerate(body, start=idx_title + 2):
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("+"):
            if logical_lines:
                prev_no, prev_text = logical_lines[-1]
                continuation = stripped[1:].strip()
                logical_lines[-1] = (prev_no, f"{prev_text} {continuation}")
            continue
        logical_lines.append((lineno, stripped))

    raw_logical = [l for _, l in logical_lines]
    expander = SubcircuitExpander()
    expanded_lines, hierarchy = expander.expand(raw_logical, base_dir=base_dir)

    elements: list[_Element] = []
    for lineno, line in enumerate(expanded_lines, start=1):
        if line.lower().startswith(".end"):
            break
        if line.startswith("."):
            continue
        el = _parse_line(line, lineno)
        if el is not None:
            elements.append(el)

    return elements, hierarchy


def parse_netlist_string(text: str, base_dir: Path | None = None) -> list[_Element]:
    """Parse a SPICE netlist from a multi-line string.

    Parameters
    ----------
    text : str
        Full netlist content as a string.
    base_dir : Path | None, optional
        Base directory for resolving .include directives.

    Returns
    -------
    list[_Element]
        Ordered list of circuit elements.
    """
    elements, _ = parse_netlist_with_hierarchy(text, base_dir=base_dir)
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
    elements, _ = parse_netlist_with_hierarchy(p, base_dir=p.parent)
    return elements


def parse_dc_directive(text_or_path: str | Path) -> dict[str, object] | None:
    """Extract .DC sweep directive from a netlist file or text string.

    Supports single sweep:
        .DC <srcnam> <vstart> <vstop> <vincr>
    And 2D nested sweep:
        .DC <src1> <start1> <stop1> <step1> <src2> <start2> <stop2> <step2>

    Returns
    -------
    dict or None
        Parsed sweep parameters dictionary, or None if no .DC line was found.
    """
    text = str(text_or_path)
    p = Path(text)
    if p.exists() and p.is_file():
        text = p.read_text(encoding="utf-8")

    for line in text.splitlines():
        line = _strip_comment(line)
        if line.upper().startswith(".DC"):
            tokens = line.split()
            if len(tokens) >= 5:
                res: dict[str, object] = {
                    "param_name": tokens[1].upper(),
                    "start": parse_value(tokens[2]),
                    "stop": parse_value(tokens[3]),
                    "step": parse_value(tokens[4]),
                }
                if len(tokens) >= 9:
                    res.update({
                        "param2_name": tokens[5].upper(),
                        "start2": parse_value(tokens[6]),
                        "stop2": parse_value(tokens[7]),
                        "step2": parse_value(tokens[8]),
                    })
                return res
    return None
