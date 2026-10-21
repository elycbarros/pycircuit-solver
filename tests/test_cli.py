"""
test_cli.py — Testes da interface de linha de comando (CLI) pycircuit.
"""

import pytest

from circuit_solver.cli import main


@pytest.fixture
def sample_netlist(tmp_path):
    p = tmp_path / "circuit.cir"
    p.write_text("""* Sample Netlist
V1 1 0 DC 10 AC 10 0
R1 1 2 1k
R2 2 0 1k
C1 2 0 1u
.end
""")
    return str(p)


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "pycircuit" in captured.out


def test_cli_solve(sample_netlist, capsys):
    ret = main(["solve", sample_netlist])
    assert ret == 0
    captured = capsys.readouterr()
    assert "MNA Solution" in captured.out
    assert "V(     1) =" in captured.out


def test_cli_thevenin(sample_netlist, capsys):
    ret = main(["thevenin", sample_netlist, "-a", "2", "-b", "0"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Thévenin / Norton Equivalent" in captured.out


def test_cli_monte_carlo(sample_netlist, capsys):
    ret = main(["monte-carlo", sample_netlist, "-n", "20"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Monte Carlo Analysis" in captured.out


def test_cli_ac_single_freq(sample_netlist, capsys):
    ret = main(["ac", sample_netlist, "-f", "100"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "AC Phasor Solution" in captured.out


def test_cli_ac_sweep(sample_netlist, capsys):
    ret = main(["ac", sample_netlist, "--sweep", "10", "1000", "5", "--node", "2"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Varredura CA concluída" in captured.out
