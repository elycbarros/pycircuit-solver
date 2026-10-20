"""
cap05_opamp_inversor.py — Amplificadores Operacionais Ideais (Alexander & Sadiku, Cap. 5)

Circuito: Amplificador Inversor Clássico
----------------------------------------

              ┌───── Rf (50kΩ) ─────┐
              │                     │
    Vin ──── R1 (10kΩ) ─── (-)      │
    (2V)                    │       ├── Vout
                     (+) ───┴───────┘
                      │
                     GND (0)

Ganho teórico:
    Av = - Rf / R1 = - 50k / 10k = -5
    Vout = Av * Vin = -5 * 2V = -10V
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import OpAmp, Resistor, VoltageSource
from circuit_solver.engine.mna import solve

# ---------------------------------------------------------------------------
# Montagem do Amplificador Inversor
# ---------------------------------------------------------------------------
elements = [
    # Fonte de entrada: 2V referenciada ao terra ("0")
    VoltageSource("Vin", "in", "0", 2.0),

    # Resistor de entrada R1 entre "in" e a entrada inversora "vn"
    Resistor("R1", "in", "vn", 10_000.0),

    # Resistor de realimentação Rf entre "vn" e a saída "vout"
    Resistor("Rf", "vn", "vout", 50_000.0),

    # Carga na saída Rload de 10kΩ para o terra
    Resistor("Rload", "vout", "0", 10_000.0),

    # OpAmp ideal: in_p="0", in_n="vn", out="vout", gnd="0"
    OpAmp("U1", in_p="0", in_n="vn", out="vout", gnd="0"),
]

circuit = Circuit(elements)
sol = solve(circuit)

print("\n" + "=" * 56)
print("  Amplificador Operacional Inversor — Cap. 5")
print("=" * 56)
print(sol.summary())

print(f"\nTensão de entrada Vin: {sol.V('in'):.2f} V")
print(f"Potencial no nó virtual vn (deve ser ~0 V): {sol.V('vn'):.6e} V")
print(f"Tensão de saída Vout: {sol.V('vout'):.4f} V (Teórico: -10.00 V)")
print(f"Ganho medido Av = Vout / Vin: {sol.V('vout') / sol.V('in'):.2f}")
print(f"Corrente fornecida pelo OpAmp na saída: {sol.I_vsource('U1') * 1e3:.4f} mA")
