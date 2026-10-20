"""
cap07_transiente_rlc.py — Resposta ao Degrau de Circuito RLC de 2ª Ordem.

Exemplo clássico de transitório:
  * Fonte de tensão com degrau de 0V para 10V em t=0
  * Resistor R = 20 Ω
  * Indutor L = 10 mH
  * Capacitor C = 1 μF

Comportamento físico esperado:
  * Frequência natural: ω0 = 10.000 rad/s
  * Coeficiente de amortecimento: α = R / (2L) = 1.000 s^-1
  * Como α < ω0: Circuito subamortecido (apresenta oscilação e overshoot característico)
  * Tensão máxima de pico: V_peak ≈ 17.3 V
  * Assentamento final: V_final = 10.0 V
"""

from __future__ import annotations

import numpy as np

from circuit_solver import (
    Circuit,
    VoltageSource,
    Resistor,
    Inductor,
    Capacitor,
    StepWaveform,
    solve_transient,
)
from circuit_solver.visualization.plots import plot_transient


def main() -> None:
    print("=" * 72)
    print("⚡ Exemplo: Análise Transitória RLC Subamortecido de 2ª Ordem")
    print("=" * 72)

    # 1. Definir circuito com fonte de degrau
    circuit = Circuit([
        VoltageSource("V1", "1", "0", 10.0, waveform=StepWaveform(v1=0.0, v2=10.0, t_step=0.0)),
        Resistor("R1", "1", "2", 20.0),
        Inductor("L1", "2", "3", 10e-3, ic=0.0),
        Capacitor("C1", "3", "0", 1e-6, ic=0.0),
    ])

    # 2. Executar simulação no tempo
    t_stop = 8e-3  # 8 milissegundos
    dt = 1e-6      # 1 microssegundo de passo

    sol = solve_transient(circuit, t_stop=t_stop, dt=dt, method="trapezoidal", uic=True)

    print(sol.summary())

    v_c = sol.V("3")
    v_peak = np.max(v_c)
    t_peak = sol.time[np.argmax(v_c)]

    print(f"\n📊 Análise de Desempenho Dinâmico:")
    print(f"  Tensão máxima de pico (Overshoot): {v_peak:.3f} V (em t = {t_peak*1000:.3f} ms)")
    print(f"  Tensão final de regime:            {v_c[-1]:.3f} V")
    print(f"  Corrente máxima no indutor:        {np.max(sol.I('L1')):.3f} A")

    # 3. Gerar gráfico das formas de onda
    save_file = "rlc_transiente.png"
    plot_transient(
        time=sol.time,
        node_voltages=sol.node_voltages,
        nodes_to_plot=["1", "3"],
        branch_currents=sol.branch_currents,
        elements_to_plot=["L1"],
        title="Resposta ao Degrau — RLC Série Subamortecido",
        show=False,
        save_path=save_file,
    )
    print(f"\nGráfico exportado com sucesso para: {save_file}")


if __name__ == "__main__":
    main()
