"""
cap04_thevenin_analise.py — Teorema de Thévenin, Norton e Máxima Transferência de Potência
(Alexander & Sadiku, Cap. 4)

Circuito com Fontes Dependentes e Independentes
-----------------------------------------------
Vamos analisar um circuito clássico para encontrar o equivalente Thévenin entre os terminais A e B:

          R1 (4Ω)         R2 (1Ω)
    1 ────────────── 2 ────────────── A (nó 3)
    │                │
   Vs (32V)         Is (2A)
    │                │
    0 ────────────── 0 ────────────── B (nó 0)
                     │
                    R3 (12Ω) entre nó 2 e 0

Parâmetros:
- Nó A = "3"
- Nó B = "0" (terra)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import CurrentSource, Resistor, VoltageSource
from circuit_solver.engine.thevenin import thevenin_norton

# ---------------------------------------------------------------------------
# Montagem do Circuito
# ---------------------------------------------------------------------------
elements = [
    VoltageSource("Vs", "1", "0", 32.0),
    Resistor("R1", "1", "2", 4.0),
    Resistor("R3", "2", "0", 12.0),
    CurrentSource("Is", "2", "0", 2.0),   # 2A entrando no nó 2 vindo do terra
    Resistor("R2", "2", "3", 1.0),        # terminal A está no nó 3
]

circuit = Circuit(elements, allowed_open_nodes=["3"])

# Extrair equivalente de Thévenin / Norton entre nós "3" (A) e "0" (B)
res = thevenin_norton(circuit, node_a="3", node_b="0")

print("\n" + "=" * 56)
print("  Teorema de Thévenin & Norton — Alexander & Sadiku Cap. 4")
print("=" * 56)
print(res.summary())

# Plot opcional
try:
    import matplotlib
    matplotlib.use("Agg")
    from circuit_solver.visualization.plots import plot_power_transfer
    fig = plot_power_transfer(res, show=False, save_path="thevenin_power_curve.png")
    print("\nCurva de Máxima Transferência de Potência salva em: thevenin_power_curve.png")
except Exception as e:
    print(f"(Visualização ignorada: {e})")
