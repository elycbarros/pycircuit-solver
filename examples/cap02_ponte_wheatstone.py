"""
cap02_ponte_wheatstone.py — Wheatstone Bridge (Alexander & Sadiku, Cap. 2)

Circuit topology
----------------

         R1 (1kΩ)       R3 (2kΩ)
    1 ──────────── 2 ──────────── 3
    │                              │
   V1 (10V)       Rg (1kΩ, galv)  │
    │              bridge          │
    0 ──────────── 4 ──────────── 3
         R2 (2kΩ)       R4 (1kΩ)

Balanced condition: R1/R2 = R3/R4  →  R1*R4 = R2*R3
Here: 1k * 1k = 1M ≠ 2k * 2k = 4M  →  unbalanced bridge.
Galvanometer branch carries a non-zero current.

Objectives
----------
1. Solve for all node voltages.
2. Print branch currents and power audit.
3. Monte Carlo: effect of ±5% resistor tolerances on V(2) and V(4).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from circuit_solver.core.elements import Resistor, VoltageSource
from circuit_solver.core.circuit import Circuit
from circuit_solver.engine.mna import solve
from circuit_solver.engine.monte_carlo import monte_carlo

# ---------------------------------------------------------------------------
# Build circuit
# ---------------------------------------------------------------------------
elements = [
    VoltageSource("V1", "1", "0", 10.0),      # 10 V supply
    Resistor("R1", "1", "2", 1000.0),          # 1 kΩ
    Resistor("R2", "1", "4", 2000.0),          # 2 kΩ
    Resistor("R3", "2", "3", 2000.0),          # 2 kΩ
    Resistor("R4", "4", "3", 1000.0),          # 1 kΩ
    Resistor("Rg", "2", "4", 1000.0),          # Galvanometer branch 1 kΩ
    Resistor("Rl", "3", "0", 1.0),             # Load / closing resistor (small)
]

circuit = Circuit(elements)
sol = solve(circuit)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 56)
print("  Ponte de Wheatstone — Resultados")
print("=" * 56)
print(sol.summary())

print("\nCorrente no galvanômetro (Rg):", f"{sol.I_resistor('Rg')*1e3:.4f} mA")
print("Tensão no galvanômetro: V(2)−V(4) =",
      f"{sol.V_diff('2','4'):.6f} V")

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
print("\n" + "─" * 56)
print("  Simulação Monte Carlo (1000 amostras, ±5 %)")
print("─" * 56)
mc_result = monte_carlo(circuit, n_samples=1000)
print(mc_result.summary())

try:
    from circuit_solver.visualization.plots import plot_monte_carlo_all
    import matplotlib
    matplotlib.use("Agg")   # non-interactive for script execution
    fig = plot_monte_carlo_all(mc_result, show=False,
                               save_path="wheatstone_mc.png")
    print("\nHistograma salvo em: wheatstone_mc.png")
except Exception as e:
    print(f"(Visualização ignorada: {e})")
