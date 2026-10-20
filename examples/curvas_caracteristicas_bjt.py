"""
curvas_caracteristicas_bjt.py — Família de Curvas Características de Transistor (IC x VCE).

Demonstra o uso da varredura paramétrica 2D (.dc sweep):
- Varredura primária: VCE de 0 a 5V com passo de 0.05V.
- Varredura secundária aninhada: IB em [10µA, 20µA, 30µA, 40µA, 50µA].
- Gera a família de curvas clássica mostrando a região de saturação e a região ativa.
"""

from pathlib import Path
from circuit_solver import (
    Circuit,
    VoltageSource,
    CurrentSource,
    BJT,
    dc_sweep,
    plot_dc_sweep,
)


def main():
    print("=" * 65)
    print("  Curvas Características de Transistor BJT (IC x VCE)")
    print("=" * 65)

    # 1. Montagem do circuito de teste
    c = Circuit([
        VoltageSource("VCE", "c", "0", 0.0),
        CurrentSource("IB", "b", "0", 10e-6),
        BJT("Q1", node_c="c", node_b="b", node_e="0", polarity="NPN", beta=100.0),
    ])

    # 2. Executa a varredura paramétrica 2D
    print("\nExecutando varredura DC 2D:")
    print("  VCE : 0.0 V a 5.0 V (passo 0.05 V)")
    print("  IB  : 10 µA a 50 µA (passo 10 µA)")

    res = dc_sweep(
        c,
        param_name="VCE",
        start=0.0,
        stop=5.0,
        step=0.05,
        param2_name="IB",
        start2=10e-6,
        stop2=50e-6,
        step2=10e-6,
    )

    print("\n" + res.summary())

    # 3. Exibe valores de saturação e corrente ativa para cada IB
    print("\nPonto em Regime Ativo (VCE = 5.0 V):")
    ic_data = res.I("Q1_C")
    for idx, ib_val in enumerate(res.sweep2_values):
        ic_active = ic_data[idx, -1] * 1e3
        print(f"  IB = {ib_val*1e6:4.0f} µA  ->  IC = {ic_active:6.3f} mA  (ganho beta ≈ {ic_active/(ib_val*1e3):.1f})")

    # 4. Gera e salva o gráfico da família de curvas
    output_png = Path(__file__).parent / "curvas_caracteristicas_bjt.png"
    plot_dc_sweep(
        res,
        branch_to_plot="Q1_C",
        title="Curvas Características do Transistor NPN (IC x VCE)",
        save_path=str(output_png),
    )
    print(f"\nGráfico salvo com sucesso em: {output_png}")


if __name__ == "__main__":
    main()
