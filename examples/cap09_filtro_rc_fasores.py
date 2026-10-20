"""
cap09_filtro_rc_fasores.py — Análise Fasorial e Resposta em Frequência (Alexander & Sadiku, Cap. 9 & 10)

Filtro Passa-Baixas RC
----------------------

    Vin (10V ∠ 0°) ──── R (1kΩ) ──── out (nó 2)
                                       │
                                     C (1μF)
                                       │
                                     0 (GND)

Frequência de corte teórica:
    fc = 1 / (2 * pi * R * C) = 1 / (2 * pi * 1000 * 1e-6) ≈ 159.15 Hz

Comportamento na frequência de corte (f = fc):
    |Vout| = Vin / sqrt(2) ≈ 7.07 V (-3 dB)
    Fase = -45°
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import Capacitor, Resistor, VoltageSource
from circuit_solver.engine.ac import solve_ac, ac_sweep

# ---------------------------------------------------------------------------
# Montagem do Circuito
# ---------------------------------------------------------------------------
R_val = 1000.0   # 1 kΩ
C_val = 1e-6     # 1 μF
fc = 1.0 / (2.0 * np.pi * R_val * C_val)

elements = [
    VoltageSource("Vin", "in", "0", 0.0, ac_mag=10.0, ac_phase=0.0),
    Resistor("R1", "in", "out", R_val),
    Capacitor("C1", "out", "0", C_val),
]

circuit = Circuit(elements)

# ---------------------------------------------------------------------------
# 1. Solução pontual na frequência de corte (fc)
# ---------------------------------------------------------------------------
sol = solve_ac(circuit, frequency=fc)

print("\n" + "=" * 64)
print(f"  Filtro Passa-Baixas RC — Solução Fasorial em fc = {fc:.2f} Hz")
print("=" * 64)
print(sol.summary())

print(f"\nTensão na saída |V(out)| : {sol.V_mag('out'):.4f} V (Teórico: {10.0/np.sqrt(2):.4f} V)")
print(f"Defasagem na saída ∠V(out): {sol.V_phase_deg('out'):.2f}° (Teórico: -45.00°)")
print(f"Atenuação: {20 * np.log10(sol.V_mag('out') / 10.0):.2f} dB")

# ---------------------------------------------------------------------------
# 2. Varredura de frequência (Bode) de 10 Hz a 100 kHz
# ---------------------------------------------------------------------------
freqs = np.logspace(1, 5, 200)
frequencies, mag_db, phase_deg = ac_sweep(circuit, freqs, target_node="out")

try:
    import matplotlib
    matplotlib.use("Agg")
    from circuit_solver.visualization.plots import plot_bode
    fig = plot_bode(frequencies, mag_db, phase_deg,
                    title=f"Resposta em Frequência (Filtro RC, fc = {fc:.1f} Hz)",
                    target_label="V(out)", show=False,
                    save_path="filtro_rc_bode.png")
    print("\nDiagrama de Bode salvo com sucesso em: filtro_rc_bode.png")
except Exception as e:
    print(f"(Visualização ignorada: {e})")
