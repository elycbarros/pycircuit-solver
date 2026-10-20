"""
cli.py — Command-Line Interface (CLI) for pycircuit-solver.

Commands
--------
* pycircuit solve <netlist.cir>
* pycircuit thevenin <netlist.cir> -a <node_a> -b <node_b> [--plot]
* pycircuit monte-carlo <netlist.cir> [-n <samples>] [-t <tol>] [--plot]
* pycircuit ac <netlist.cir> -f <freq> [--sweep <start> <stop> <points>] [--node <node>] [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.engine.mna import solve
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.engine.monte_carlo import monte_carlo
from circuit_solver.engine.ac import solve_ac, ac_sweep
from circuit_solver.io.parser import parse_netlist
from circuit_solver.visualization.plots import (
    plot_power_transfer,
    plot_monte_carlo_all,
    plot_bode,
)


def _cmd_solve(args: argparse.Namespace) -> int:
    elements = parse_netlist(args.netlist)
    circuit = Circuit(elements)
    solution = solve(circuit)
    print(solution.summary())
    return 0


def _cmd_thevenin(args: argparse.Namespace) -> int:
    elements = parse_netlist(args.netlist)
    circuit = Circuit(elements, allowed_open_nodes=[args.node_a, args.node_b])
    res = thevenin_norton(circuit, node_a=args.node_a, node_b=args.node_b)
    print(res.summary())

    if args.plot or args.save:
        plot_power_transfer(
            res,
            show=args.plot,
            save_path=args.save,
        )
        if args.save:
            print(f"Gráfico de Thévenin salvo em: {args.save}")
    return 0


def _cmd_monte_carlo(args: argparse.Namespace) -> int:
    elements = parse_netlist(args.netlist)
    circuit = Circuit(elements)
    res = monte_carlo(
        circuit,
        n_samples=args.samples,
        seed=args.seed,
        override_tolerance=args.tolerance,
    )
    print(res.summary())

    if args.plot or args.save:
        plot_monte_carlo_all(
            res,
            show=args.plot,
            save_path=args.save,
        )
        if args.save:
            print(f"Histogramas de Monte Carlo salvos em: {args.save}")
    return 0


def _cmd_ac(args: argparse.Namespace) -> int:
    elements = parse_netlist(args.netlist)
    circuit = Circuit(elements)

    if args.sweep:
        f_start, f_stop, n_points = args.sweep
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), int(n_points))
        target_node = args.node or circuit.nodes[0]
        frequencies, mag_db, phase_deg = ac_sweep(circuit, freqs, target_node)

        print(f"\nVarredura CA concluída para o nó '{target_node}' ({len(freqs)} pontos).")
        print(f"Frequência inicial: {f_start:.2f} Hz | Frequência final: {f_stop:.2f} Hz")
        print(f"Pico de magnitude: {np.max(mag_db):.2f} dB | Mínimo: {np.min(mag_db):.2f} dB")

        if args.plot or args.save:
            plot_bode(
                frequencies,
                mag_db,
                phase_deg,
                target_label=target_node,
                show=args.plot,
                save_path=args.save,
            )
            if args.save:
                print(f"Diagrama de Bode salvo em: {args.save}")
    else:
        freq = args.frequency if args.frequency is not None else 60.0
        sol = solve_ac(circuit, frequency=freq)
        print(sol.summary())

    return 0


def _cmd_tran(args: argparse.Namespace) -> int:
    from circuit_solver.engine.transient import solve_transient
    from circuit_solver.visualization.plots import plot_transient

    elements = parse_netlist(args.netlist)
    circuit = Circuit(elements)

    t_stop = args.tstop
    dt = args.dt if args.dt is not None else (t_stop / 1000.0)

    sol = solve_transient(
        circuit,
        t_stop=t_stop,
        dt=dt,
        method=args.method,
        uic=args.uic,
    )
    print(sol.summary())

    if args.plot or args.save:
        nodes = [args.node] if args.node else None
        plot_transient(
            time=sol.time,
            node_voltages=sol.node_voltages,
            nodes_to_plot=nodes,
            branch_currents=sol.branch_currents,
            elements_to_plot=[args.element] if args.element else None,
            title=f"Transient Analysis (t_stop = {t_stop:.2e} s)",
            show=args.plot,
            save_path=args.save,
        )
        if args.save:
            print(f"Gráfico de transiente salvo em: {args.save}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pycircuit",
        description="⚡ pycircuit-solver: Análise Nodal Modificada (MNA) DC, CA e Transitória",
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")

    # solve
    p_solve = subparsers.add_parser("solve", help="Resolver circuito DC a partir de netlist")
    p_solve.add_argument("netlist", type=str, help="Caminho do arquivo netlist SPICE (.cir)")

    # thevenin
    p_th = subparsers.add_parser("thevenin", help="Extrair equivalente Thévenin/Norton e curva P_max")
    p_th.add_argument("netlist", type=str, help="Caminho do arquivo netlist SPICE (.cir)")
    p_th.add_argument("-a", "--node-a", required=True, type=str, help="Terminal A (positivo)")
    p_th.add_argument("-b", "--node-b", default="0", type=str, help="Terminal B (referência, padrão '0')")
    p_th.add_argument("--plot", action="store_true", help="Exibir gráfico da curva de potência")
    p_th.add_argument("--save", type=str, default=None, help="Salvar gráfico da curva de potência em arquivo")

    # monte-carlo
    p_mc = subparsers.add_parser("monte-carlo", help="Análise estatística de tolerâncias de resistores")
    p_mc.add_argument("netlist", type=str, help="Caminho do arquivo netlist SPICE (.cir)")
    p_mc.add_argument("-n", "--samples", type=int, default=1000, help="Número de amostras (padrão 1000)")
    p_mc.add_argument("-s", "--seed", type=int, default=42, help="Seed para reproduzibilidade estatística (padrão 42)")
    p_mc.add_argument("-t", "--tolerance", type=float, default=None, help="Sobrescrever tolerância (ex: 0.05 para ±5%%)")
    p_mc.add_argument("--plot", action="store_true", help="Exibir histogramas de distribuição")
    p_mc.add_argument("--save", type=str, default=None, help="Salvar histogramas em arquivo")

    # ac
    p_ac = subparsers.add_parser("ac", help="Análise fasorial senoidal em regime permanente (CA)")
    p_ac.add_argument("netlist", type=str, help="Caminho do arquivo netlist SPICE (.cir)")
    p_ac.add_argument("-f", "--frequency", type=float, default=60.0, help="Frequência em Hz (padrão 60 Hz)")
    p_ac.add_argument("--sweep", nargs=3, type=float, metavar=("START", "STOP", "POINTS"),
                      help="Varredura de frequência em escala logarítmica: START STOP POINTS")
    p_ac.add_argument("--node", type=str, default=None, help="Nó para medição na varredura")
    p_ac.add_argument("--plot", action="store_true", help="Exibir diagrama de Bode")
    p_ac.add_argument("--save", type=str, default=None, help="Salvar diagrama de Bode em arquivo")

    # tran
    p_tran = subparsers.add_parser("tran", help="Análise transitória no domínio do tempo (carga/descarga RC, RL, RLC)")
    p_tran.add_argument("netlist", type=str, help="Caminho do arquivo netlist SPICE (.cir)")
    p_tran.add_argument("--tstop", type=float, required=True, help="Tempo final de simulação em segundos (obrigatório)")
    p_tran.add_argument("--dt", type=float, default=None, help="Passo de tempo em segundos (padrão tstop/1000)")
    p_tran.add_argument("--method", choices=["trapezoidal", "euler"], default="trapezoidal",
                        help="Método de integração numérica (padrão 'trapezoidal')")
    p_tran.add_argument("--node", type=str, default=None, help="Nó específico para plotar")
    p_tran.add_argument("--element", type=str, default=None, help="Elemento para plotar corrente")
    p_tran.add_argument("--uic", action="store_true", help="Usar condições iniciais dos elementos (UIC)")
    p_tran.add_argument("--plot", action="store_true", help="Exibir gráfico das formas de onda")
    p_tran.add_argument("--save", type=str, default=None, help="Salvar gráfico das formas de onda em arquivo")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "solve":
            return _cmd_solve(args)
        elif args.command == "thevenin":
            return _cmd_thevenin(args)
        elif args.command == "monte-carlo":
            return _cmd_monte_carlo(args)
        elif args.command == "ac":
            return _cmd_ac(args)
        elif args.command == "tran":
            return _cmd_tran(args)
        else:
            parser.print_help()
            return 1
    except Exception as exc:
        print(f"\n❌ Erro na execução: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
