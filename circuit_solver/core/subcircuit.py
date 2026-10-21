"""
subcircuit.py — Hierarchical subcircuit definition, instantiation, and expansion.

This module implements industrial SPICE-standard subcircuit processing
(.SUBCKT / .ENDS and 'X' element instances) combined with an academic
"White-Box" pedagogical introspection engine designed for engineering
education, dissertations, and research.

Features:
---------
* Full SPICE-compatible .SUBCKT / .ENDS syntax with positional port mapping.
* Parameterization via 'PARAMS: key=val' or 'key=val' with mathematical expressions ({...}).
* Nested subcircuit instantiation (subcircuits within subcircuits).
* Scoped node qualification (e.g. 'X1.int_node') and element qualification ('X1.R1').
* Global ground node '0' is never qualified.
* Pedagogical inspection:
  - Instance tree visualizer (hierarchy_tree).
  - Step-by-step expansion and pin mapping report (explain).
  - Dimensional MNA matrix impact analysis (added nodes and auxiliary variables).
  - Scoped variable probing in solutions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from circuit_solver.core.validator import parse_value


# ---------------------------------------------------------------------------
# Safe Mathematical Expression Evaluator for SPICE PARAMS
# ---------------------------------------------------------------------------

_SAFE_MATH_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
}


def tokenize_spice_line(line: str) -> list[str]:
    """Tokenize a SPICE line respecting curly brace expressions {...} and quotes."""
    return re.findall(r"\{[^{}]*\}|\"[^\"]*\"|'[^']*'|\S+", line)


def evaluate_param_expression(expr: str | float | int, scope: dict[str, float]) -> float:
    """Evaluate a numeric value or SPICE mathematical expression with parameter substitution.

    Parameters
    ----------
    expr : str | float | int
        Value or expression string (e.g. "10k", "{R1 * 2}", "{1 / (2*pi*FC*C)}").
    scope : dict[str, float]
        Dictionary of available variable names and their numeric values.

    Returns
    -------
    float
        The computed floating-point value.
    """
    if isinstance(expr, (int, float)):
        return float(expr)

    s = str(expr).strip()
    had_braces = s.startswith("{") and s.endswith("}")
    if had_braces:
        s = s[1:-1].strip()

    has_operators = any(op in s for op in ("+", "-", "*", "/", "^", "(", ")"))

    if not has_operators and not had_braces:
        if " " not in s:
            try:
                return parse_value(s)
            except Exception:
                pass

    # If it directly matches a variable name in scope (case-insensitive)
    for k, v in scope.items():
        if s.upper() == k.upper():
            return float(v)

    # Replace engineering values and scope variables inside the expression
    # 1. Substitute scope variables
    # Sort by length descending so that longer variable names get replaced first
    sorted_vars = sorted(scope.keys(), key=len, reverse=True)
    eval_str = s
    for var in sorted_vars:
        pattern = r"\b" + re.escape(var) + r"\b"
        eval_str = re.sub(pattern, f"({scope[var]})", eval_str, flags=re.IGNORECASE)

    # 2. Replace remaining engineering suffixes on literals (e.g. 10k -> 10000.0)
    def _sub_eng(match: re.Match) -> str:
        tok = match.group(0)
        try:
            float(tok)
            return tok
        except ValueError:
            pass
        try:
            return f"({parse_value(tok)})"
        except Exception:
            return tok

    eval_str = re.sub(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:meg|MEG|[a-df-zA-DF-Z]+)\b", _sub_eng, eval_str)

    # Replace caret ^ with ** for power
    eval_str = eval_str.replace("^", "**")

    eval_env = dict(_SAFE_MATH_NAMES)
    eval_env.update(scope)

    try:
        val = eval(eval_str, {"__builtins__": None}, eval_env)
        return float(val)
    except Exception as exc:
        raise ValueError(f"Failed to evaluate SPICE expression '{expr}': {exc}") from exc


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SubcircuitDefinition:
    """Represents a .SUBCKT template definition.

    Parameters
    ----------
    name : str
        Subcircuit model name in uppercase (e.g. 'OPAMP_741', 'RC_FILTER').
    pins : list[str]
        Ordered formal terminal pin names (e.g. ['INP', 'INN', 'OUT', 'VCC', 'VEE']).
    default_params : dict[str, str | float]
        Default parameters specified in the .SUBCKT header via PARAMS:.
    lines : list[str]
        Raw netlist lines comprising the internal circuit.
    """

    name: str
    pins: list[str]
    default_params: dict[str, str | float] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


@dataclass
class SubcircuitInstance:
    """Represents an instantiated subcircuit block ('X' element).

    Stores pedagogical metadata allowing full inspection of how the subcircuit
    was wired, parameterized, and expanded into the global system.
    """

    name: str
    subckt_name: str
    connected_nodes: list[str]
    param_overrides: dict[str, str | float] = field(default_factory=dict)
    resolved_params: dict[str, float] = field(default_factory=dict)
    pin_mapping: dict[str, str] = field(default_factory=dict)
    internal_nodes: list[str] = field(default_factory=list)
    element_names: list[str] = field(default_factory=list)
    child_instances: list["SubcircuitInstance"] = field(default_factory=list)

    def tree_summary(self, indent: int = 0) -> str:
        """Return an indented ASCII tree representation of this instance."""
        prefix = "  " * indent
        conns = ", ".join(f"{p}→{n}" for p, n in self.pin_mapping.items())
        params = ", ".join(f"{k}={v:g}" for k, v in self.resolved_params.items())
        p_str = f" [PARAMS: {params}]" if params else ""
        lines = [
            f"{prefix}└── {self.name} ({self.subckt_name}){p_str}",
            f"{prefix}    Conexões: {conns}",
            f"{prefix}    Nós Internos: {', '.join(self.internal_nodes) if self.internal_nodes else '(nenhum)'}",
            f"{prefix}    Elementos: {', '.join(self.element_names)}",
        ]
        for child in self.child_instances:
            lines.append(child.tree_summary(indent + 2))
        return "\n".join(lines)


class HierarchyTree:
    """Manages the full hierarchical tree of subcircuit definitions and instances."""

    def __init__(self) -> None:
        self.definitions: dict[str, SubcircuitDefinition] = {}
        self.root_instances: list[SubcircuitInstance] = []

    def add_definition(self, subckt: SubcircuitDefinition) -> None:
        """Register a subcircuit definition."""
        self.definitions[subckt.name.upper()] = subckt

    def add_instance(self, inst: SubcircuitInstance) -> None:
        """Register a top-level subcircuit instance."""
        self.root_instances.append(inst)

    def all_instances(self) -> list[SubcircuitInstance]:
        """Return a flattened list of all instances (including nested ones)."""
        result: list[SubcircuitInstance] = []

        def _traverse(inst: SubcircuitInstance) -> None:
            result.append(inst)
            for child in inst.child_instances:
                _traverse(child)

        for root in self.root_instances:
            _traverse(root)
        return result

    def find_instance(self, name: str) -> SubcircuitInstance | None:
        """Find an instance by name (e.g. 'X1' or 'X1.X_AMP')."""
        target = name.upper()
        for inst in self.all_instances():
            if inst.name.upper() == target:
                return inst
        return None

    def tree_view(self) -> str:
        """Render a clean ASCII tree of the circuit hierarchy."""
        if not self.root_instances:
            return "Circuito Plano (Sem subcircuitos instanciados)."
        lines = ["Hierarquia de Subcircuitos:"]
        for root in self.root_instances:
            lines.append(root.tree_summary(indent=0))
        return "\n".join(lines)

    def explain(self, circuit: Any | None = None) -> str:
        """Generate a comprehensive pedagogical report explaining the expansion.

        This report provides complete transparency for students, educators, and
        researchers on how the hierarchical subcircuits were transformed into
        the Modified Nodal Analysis (MNA) system.
        """
        all_insts = self.all_instances()
        if not all_insts:
            return "Nenhum subcircuito (.SUBCKT / X) presente neste circuito."

        lines = [
            "=" * 78,
            " 🎓 RELATÓRIO PEDAGÓGICO DE EXPANSÃO HIERÁRQUICA (WHITE-BOX SUBCIRCUITS)",
            "=" * 78,
            "",
            "1. ESTRUTURA HIERÁRQUICA DO CIRCUITO:",
            self.tree_view(),
            "",
            "2. DETALHAMENTO DE INSTANCIAÇÃO E DE-PARA DE PINOS:",
        ]

        total_internal_nodes = 0
        total_elements = 0

        for inst in all_insts:
            total_internal_nodes += len(inst.internal_nodes)
            total_elements += len(inst.element_names)

            lines.append(f"  • Instância: [{inst.name}] do modelo <{inst.subckt_name}>")
            lines.append("    Mapeamento de Pinos Formais -> Nós do Circuito:")
            for pin, node in inst.pin_mapping.items():
                scope_info = " (terra/referência)" if node == "0" else ""
                lines.append(f"      - Pino '{pin}'  ===>  Nó '{node}'{scope_info}")

            if inst.resolved_params:
                lines.append("    Parâmetros Resolvidos:")
                for p_name, p_val in inst.resolved_params.items():
                    lines.append(f"      - {p_name} = {p_val:g}")

            if inst.internal_nodes:
                lines.append(f"    Nós Internos Privados Gerados ({len(inst.internal_nodes)}):")
                lines.append(f"      {', '.join(inst.internal_nodes)}")
            else:
                lines.append("    Nenhum nó interno gerado (todos os nós conectados aos pinos externos).")

            lines.append(f"    Elementos Expandidos ({len(inst.element_names)}):")
            lines.append(f"      {', '.join(inst.element_names)}")
            lines.append("")

        lines.extend([
            "3. IMPACTO DIMENSIONAL NA MATRIZ MNA:",
            f"  • Total de instâncias de subcircuitos: {len(all_insts)}",
            f"  • Total de elementos introduzidos: {total_elements}",
            f"  • Total de nós internos privados adicionados: {total_internal_nodes}",
        ])

        if circuit is not None and hasattr(circuit, "n_nodes") and hasattr(circuit, "size"):
            lines.extend([
                f"  • Total final de nós ativos do sistema: {circuit.n_nodes}",
                f"  • Variáveis de corrente auxiliar MNA: {circuit.n_aux}",
                f"  • Dimensão total do sistema de equações MNA: {circuit.size} x {circuit.size}",
            ])

        lines.extend([
            "",
            "💡 NOTA PEDAGÓGICA:",
            "  O simulador realizou o mapeamento 'White-Box', garantindo que",
            "  todas as grandezas internas continuem acessíveis para medição",
            "  através de sol.V('X_instancia.no') ou sol.subcircuit_summary('X_instancia').",
            "=" * 78,
        ])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hierarchical Subcircuit Expander
# ---------------------------------------------------------------------------

class SubcircuitExpander:
    """Expands hierarchical SPICE netlists into a flat netlist with full traceability."""

    def __init__(self) -> None:
        self.hierarchy = HierarchyTree()

    def load_builtin_library(self) -> None:
        """Register built-in academic and standard subcircuit models."""
        from circuit_solver.core.builtin_library import BUILTIN_SUBCIRCUITS
        for sub in BUILTIN_SUBCIRCUITS:
            self.hierarchy.add_definition(sub)

    def parse_header_params(self, param_tokens: Sequence[str]) -> dict[str, str]:
        """Extract key=value parameters from a list of tokens following PARAMS: or direct."""
        params: dict[str, str] = {}
        for token in param_tokens:
            if token.upper() == "PARAMS:":
                continue
            if "=" in token:
                k, v = token.split("=", 1)
                params[k.strip().upper()] = v.strip()
        return params

    def expand(self, raw_lines: Sequence[str], base_dir: Path | None = None) -> tuple[list[str], HierarchyTree]:
        """Process netlist lines, register .subckt blocks, and expand all 'X' instances.

        Parameters
        ----------
        raw_lines : Sequence[str]
            Lines of the netlist (with continuation lines already merged).
        base_dir : Path | None, optional
            Base directory for resolving .include directives.

        Returns
        -------
        tuple[list[str], HierarchyTree]
            Flattened element lines, and the populated HierarchyTree.
        """
        self.load_builtin_library()

        # Step 1: Collect definitions, resolve .include, and separate top-level lines
        top_lines: list[str] = []
        all_input_lines: list[str] = list(raw_lines)

        # Process .include / .inc recursively
        expanded_input_lines: list[str] = []
        for line in all_input_lines:
            stripped = line.strip()
            u = stripped.upper()
            if u.startswith(".INCLUDE") or u.startswith(".INC"):
                parts = stripped.split(maxsplit=1)
                if len(parts) > 1:
                    inc_path_str = parts[1].strip().strip('"').strip("'")
                    inc_path = Path(inc_path_str)
                    if not inc_path.is_absolute() and base_dir is not None:
                        inc_path = base_dir / inc_path
                    if inc_path.exists():
                        inc_text = inc_path.read_text(encoding="utf-8")
                        expanded_input_lines.extend(inc_text.splitlines())
                    else:
                        import warnings
                        warnings.warn(f"Included file '{inc_path}' not found — skipping.")
                continue
            expanded_input_lines.append(line)

        # Extract .SUBCKT ... .ENDS blocks
        idx = 0
        while idx < len(expanded_input_lines):
            line = expanded_input_lines[idx].strip()
            if not line or line.startswith("*"):
                idx += 1
                continue

            u = line.upper()
            if u.startswith(".SUBCKT"):
                tokens = tokenize_spice_line(line)
                if len(tokens) < 2:
                    raise SyntaxError(f"Malformed .SUBCKT line: {line}")
                sub_name = tokens[1].upper()

                # Separate pins and PARAMS
                pins: list[str] = []
                param_tokens: list[str] = []
                in_params = False

                for tok in tokens[2:]:
                    if tok.upper() == "PARAMS:" or in_params or "=" in tok:
                        in_params = True
                        param_tokens.append(tok)
                    else:
                        pins.append(tok.lower())

                default_params = self.parse_header_params(param_tokens)
                sub_body: list[str] = []
                idx += 1

                while idx < len(expanded_input_lines):
                    b_line = expanded_input_lines[idx].strip()
                    if b_line.upper().startswith(".ENDS"):
                        break
                    sub_body.append(b_line)
                    idx += 1

                sub_def = SubcircuitDefinition(
                    name=sub_name,
                    pins=pins,
                    default_params=default_params,
                    lines=sub_body,
                )
                self.hierarchy.add_definition(sub_def)
                idx += 1
                continue

            top_lines.append(line)
            idx += 1

        # Step 2: Recursively expand instances
        flat_lines: list[str] = []
        for line in top_lines:
            if line.strip().upper().startswith("X"):
                inst_lines, inst = self._expand_instance(line, parent_prefix="", scope={})
                flat_lines.extend(inst_lines)
                self.hierarchy.add_instance(inst)
            else:
                flat_lines.append(line)

        return flat_lines, self.hierarchy

    def _expand_instance(
        self,
        line: str,
        parent_prefix: str,
        scope: dict[str, float],
    ) -> tuple[list[str], SubcircuitInstance]:
        """Expand a single 'X' line into flattened element lines."""
        tokens = tokenize_spice_line(line)
        raw_name = tokens[0].upper()
        inst_name = f"{parent_prefix}{raw_name}" if parent_prefix else raw_name

        # Find the subcircuit name: it's the token right after pins
        # Collect tokens, check for PARAMS:
        node_and_model_tokens: list[str] = []
        param_tokens: list[str] = []
        in_params = False

        for tok in tokens[1:]:
            if tok.upper() == "PARAMS:" or in_params or "=" in tok:
                in_params = True
                param_tokens.append(tok)
            else:
                node_and_model_tokens.append(tok)

        if len(node_and_model_tokens) < 2:
            raise SyntaxError(f"Instance '{line}' has no subcircuit model specified.")

        sub_model_name = node_and_model_tokens[-1].upper()
        conn_nodes = [t.lower() for t in node_and_model_tokens[:-1]]

        if sub_model_name not in self.hierarchy.definitions:
            raise KeyError(
                f"Subcircuit model '{sub_model_name}' used by instance '{inst_name}' is not defined."
            )

        sub_def = self.hierarchy.definitions[sub_model_name]

        if len(conn_nodes) != len(sub_def.pins):
            raise ValueError(
                f"Instance '{inst_name}' connected to {len(conn_nodes)} nodes ({conn_nodes}), "
                f"but model '{sub_model_name}' expects {len(sub_def.pins)} pins ({sub_def.pins})."
            )

        # Pin mapping: formal_pin -> connected_node
        pin_map: dict[str, str] = {}
        for formal_pin, conn_node in zip(sub_def.pins, conn_nodes):
            pin_map[formal_pin.lower()] = conn_node

        # Resolve parameters: inherit scope, evaluate instance arguments in caller's scope
        instance_param_strs = self.parse_header_params(param_tokens)
        evaluated_instance_args: dict[str, float] = {}
        for k, v in instance_param_strs.items():
            evaluated_instance_args[k.upper()] = evaluate_param_expression(v, scope)

        resolved_params: dict[str, float] = dict(scope)

        # Evaluate defaults for parameters not overridden by instance
        for k, v in sub_def.default_params.items():
            k_u = k.upper()
            if k_u not in evaluated_instance_args:
                resolved_params[k_u] = evaluate_param_expression(v, resolved_params)

        # Apply evaluated instance overrides
        resolved_params.update(evaluated_instance_args)

        instance = SubcircuitInstance(
            name=inst_name,
            subckt_name=sub_model_name,
            connected_nodes=conn_nodes,
            param_overrides=instance_param_strs,
            resolved_params=resolved_params,
            pin_mapping=pin_map,
        )

        expanded_lines: list[str] = []
        internal_nodes_set: set[str] = set()
        element_names: list[str] = []

        def qualify_node(n: str) -> str:
            n_clean = n.strip().lower()
            if n_clean == "0":
                return "0"
            if n_clean in pin_map:
                return pin_map[n_clean]
            # Internal node: qualify with instance name prefix
            qual = f"{inst_name.lower()}.{n_clean}"
            internal_nodes_set.add(qual)
            return qual

        for sub_line in sub_def.lines:
            s_line = sub_line.strip()
            if not s_line or s_line.startswith("*") or s_line.startswith("."):
                continue

            sub_tokens = tokenize_spice_line(s_line)
            desig = sub_tokens[0].upper()
            prefix = desig[0]

            # Nested instance
            if prefix == "X":
                child_desig = sub_tokens[0]
                child_nodes_and_model: list[str] = []
                child_params: list[str] = []
                in_child_params = False
                for tok in sub_tokens[1:]:
                    if tok.upper() == "PARAMS:" or in_child_params or "=" in tok:
                        in_child_params = True
                        child_params.append(tok)
                    else:
                        child_nodes_and_model.append(tok)

                child_model = child_nodes_and_model[-1]
                child_raw_nodes = child_nodes_and_model[:-1]
                child_qual_nodes = [qualify_node(n) for n in child_raw_nodes]

                remapped_child_line = f"{child_desig} " + " ".join(child_qual_nodes) + f" {child_model}"
                if child_params:
                    remapped_child_line += " " + " ".join(child_params)

                child_lines, child_inst = self._expand_instance(
                    remapped_child_line, parent_prefix=f"{inst_name}.", scope=resolved_params
                )
                expanded_lines.extend(child_lines)
                instance.child_instances.append(child_inst)
                element_names.append(child_inst.name)
                continue

            # Standard element
            qualified_elem_name = f"{inst_name}.{desig}"
            element_names.append(qualified_elem_name)

            # Handle nodes and values per element type
            if prefix in ("R", "C", "L"):
                # Rname n+ n- val
                np = qualify_node(sub_tokens[1])
                nn = qualify_node(sub_tokens[2])
                val_tok = sub_tokens[3]
                val_num = evaluate_param_expression(val_tok, resolved_params)
                extra = " ".join(sub_tokens[4:])
                sub_line_flat = f"{qualified_elem_name} {np} {nn} {val_num:g} {extra}".strip()
                expanded_lines.append(sub_line_flat)

            elif prefix in ("V", "I"):
                # Vname n+ n- [DC val] ...
                np = qualify_node(sub_tokens[1])
                nn = qualify_node(sub_tokens[2])
                rest: list[str] = []
                idx_tok = 3
                while idx_tok < len(sub_tokens):
                    t = sub_tokens[idx_tok]
                    if t.startswith("{") or t.upper() in resolved_params:
                        v_eval = evaluate_param_expression(t, resolved_params)
                        rest.append(f"{v_eval:g}")
                    else:
                        rest.append(t)
                    idx_tok += 1
                sub_line_flat = f"{qualified_elem_name} {np} {nn} " + " ".join(rest)
                expanded_lines.append(sub_line_flat.strip())

            elif prefix in ("E", "G"):
                # Ename n+ n- nc+ nc- gain
                np = qualify_node(sub_tokens[1])
                nn = qualify_node(sub_tokens[2])
                cp = qualify_node(sub_tokens[3])
                cn = qualify_node(sub_tokens[4])
                gain_tok = sub_tokens[5]
                gain_num = evaluate_param_expression(gain_tok, resolved_params)
                expanded_lines.append(f"{qualified_elem_name} {np} {nn} {cp} {cn} {gain_num:g}")

            elif prefix in ("F", "H"):
                # Fname n+ n- Vctrl gain
                np = qualify_node(sub_tokens[1])
                nn = qualify_node(sub_tokens[2])
                vctrl = f"{inst_name}.{sub_tokens[3].upper()}"
                gain_tok = sub_tokens[4]
                gain_num = evaluate_param_expression(gain_tok, resolved_params)
                expanded_lines.append(f"{qualified_elem_name} {np} {nn} {vctrl} {gain_num:g}")

            elif prefix in ("A", "U"):
                # OpAmp: Aname in+ in- out [gnd]
                inp = qualify_node(sub_tokens[1])
                inn = qualify_node(sub_tokens[2])
                out = qualify_node(sub_tokens[3])
                gnd = qualify_node(sub_tokens[4]) if len(sub_tokens) > 4 else "0"
                expanded_lines.append(f"{qualified_elem_name} {inp} {inn} {out} {gnd}")

            elif prefix == "D":
                # Diode: Dname n+ n- ...
                np = qualify_node(sub_tokens[1])
                nn = qualify_node(sub_tokens[2])
                extra = " ".join(sub_tokens[3:])
                expanded_lines.append(f"{qualified_elem_name} {np} {nn} {extra}".strip())

            elif prefix == "Z":
                # Zener: Zname n+ n- Vz ...
                np = qualify_node(sub_tokens[1])
                nn = qualify_node(sub_tokens[2])
                vz = evaluate_param_expression(sub_tokens[3], resolved_params)
                extra = " ".join(sub_tokens[4:])
                expanded_lines.append(f"{qualified_elem_name} {np} {nn} {vz:g} {extra}".strip())

            elif prefix == "Q":
                # BJT: Qname nc nb ne ...
                nc = qualify_node(sub_tokens[1])
                nb = qualify_node(sub_tokens[2])
                ne = qualify_node(sub_tokens[3])
                extra = " ".join(sub_tokens[4:])
                expanded_lines.append(f"{qualified_elem_name} {nc} {nb} {ne} {extra}".strip())

            elif prefix == "M":
                # MOSFET: Mname nd ng ns ...
                nd = qualify_node(sub_tokens[1])
                ng = qualify_node(sub_tokens[2])
                ns = qualify_node(sub_tokens[3])
                extra = " ".join(sub_tokens[4:])
                expanded_lines.append(f"{qualified_elem_name} {nd} {ng} {ns} {extra}".strip())

            elif prefix == "K":
                # Kname L1 L2 coupling
                l1 = f"{inst_name}.{sub_tokens[1].upper()}"
                l2 = f"{inst_name}.{sub_tokens[2].upper()}"
                k_val = evaluate_param_expression(sub_tokens[3], resolved_params) if len(sub_tokens) > 3 else 0.999
                expanded_lines.append(f"{qualified_elem_name} {l1} {l2} {k_val:g}")

            elif prefix == "T":
                # Tname p1 n1 p2 n2 L1 L2 [coupling]
                p1 = qualify_node(sub_tokens[1])
                n1 = qualify_node(sub_tokens[2])
                p2 = qualify_node(sub_tokens[3])
                n2 = qualify_node(sub_tokens[4])
                l1 = evaluate_param_expression(sub_tokens[5], resolved_params)
                l2 = evaluate_param_expression(sub_tokens[6], resolved_params)
                k_val = evaluate_param_expression(sub_tokens[7], resolved_params) if len(sub_tokens) > 7 else 0.999
                expanded_lines.append(f"{qualified_elem_name} {p1} {n1} {p2} {n2} {l1:g} {l2:g} {k_val:g}")

            else:
                # Other elements: qualify terminals assuming tokens[1] and [2] are nodes
                terms = [qualify_node(t) for t in sub_tokens[1:3]]
                expanded_lines.append(f"{qualified_elem_name} " + " ".join(terms) + " " + " ".join(sub_tokens[3:]))

        instance.internal_nodes = sorted(list(internal_nodes_set))
        instance.element_names = element_names

        return expanded_lines, instance
