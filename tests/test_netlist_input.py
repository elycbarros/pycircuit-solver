"""Regressão: netlist passada como texto não pode ser tratada como caminho.

Path.exists() levanta OSError (ENAMETOOLONG) com textos longos em Python 3.10-3.12,
o que quebrava o CI nessas versões. Estes testes garantem que o texto é lido
como netlist, e que caminhos reais continuam funcionando.
"""

from pathlib import Path

from circuit_solver import Circuit, solve
from circuit_solver.io.parser import parse_dc_directive, parse_netlist_with_hierarchy


LONG_NETLIST = (
    "\n    Netlist longa para testar entrada por texto\n"
    + "".join(f"* comentario de preenchimento {i:04d} " + "x" * 80 + "\n" for i in range(40))
    + "    V1 1 0 DC 10\n"
    + "    R1 1 2 1k\n"
    + "    R2 2 0 1k\n"
    + "    .END\n"
)


def test_long_netlist_text_is_not_treated_as_path():
    assert len(LONG_NETLIST) > 255  # acima do limite típico de nome de arquivo
    elements, _ = parse_netlist_with_hierarchy(LONG_NETLIST)
    assert len(elements) == 3


def test_long_netlist_text_solves_correctly():
    sol = solve(Circuit.from_netlist(LONG_NETLIST))
    assert abs(sol.voltages["2"] - 5.0) < 1e-9


def test_short_text_without_newline_does_not_raise_oserror():
    # Uma única linha é, por convenção SPICE, só o título: não há elementos.
    # O ponto do teste é que a chamada não estoura OSError ao testar se é arquivo.
    elements, _ = parse_netlist_with_hierarchy("R1 1 0 1k")
    assert elements == []


def test_real_file_path_still_works(tmp_path: Path):
    f = tmp_path / "divisor.cir"
    f.write_text("Divisor\nV1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 1k\n.END\n", encoding="utf-8")
    sol = solve(Circuit.from_netlist(f))
    assert abs(sol.voltages["2"] - 5.0) < 1e-9


def test_dc_directive_from_long_text():
    text = LONG_NETLIST + ".DC V1 0 10 1\n"
    directive = parse_dc_directive(text)
    assert directive is not None
