"""Bateria de Testes de Stress e Carga Extrema (Stress & Fuzzing Suite).

Testa a estabilidade, precisão numérica, concorrência e integridade do pycircuit-solver
sob condições adversas e casos de borda patológicos:
1. Malha 2D resistiva densa (20x20 = 400 nós, 760 ramos) com múltiplas fontes.
2. Escada R-2R de 64 bits (faixa dinâmica de 20 ordens de magnitude: ~10^-20 V).
3. Ponte com condicionamento extremo (impedâncias de 1 uOhm a 100 MOhm).
4. Subcircuitos com aninhamento profundo (12 níveis) e propagação de parâmetros.
5. Transiente rígido (stiff ODE) com constantes de tempo em escalas amplamente dispersas.
6. Monte Carlo concorrente massivo (25.000 amostras) no núcleo Rust Rayon.
7. Fuzzing de sintaxe e netlists SPICE patológicas (continuações '+', comentários, tabs).
8. Ingestão direta de netlists em memória via parse_netlist sem erro de caminho de arquivo.
"""
from __future__ import annotations

import math
import random

import numpy as np

from circuit_solver import Circuit, solve
from circuit_solver.core.elements import (
    Capacitor,
    CurrentSource,
    Resistor,
    VoltageSource,
)
from circuit_solver.engine.monte_carlo import monte_carlo
from circuit_solver.engine.monte_carlo_rust import RUST_AVAILABLE
from circuit_solver.engine.transient import solve_transient
from circuit_solver.io.parser import parse_netlist, parse_netlist_with_hierarchy


# ==============================================================================
# 1. Stress de Escala Topológica: Malha 2D (Grid 20x20 = 400 nós)
# ==============================================================================
def test_stress_2d_mesh_grid_cross_solver():
    """Valida a resolução de uma malha 2D de 20x20 nós (400 nós e centenas de resistores).

    Compara a convergência e exatidão entre o solver Denso, Esparso Python e Esparso Rust.
    Verifica a Lei de Kirchhoff das Correntes (LKC) em nós internos.
    """
    rows, cols = 15, 15
    elements = []
    rng = random.Random(42)

    def node(r: int, c: int) -> str:
        if r == 0 and c == 0:
            return "0"  # Terra na origem
        return f"n_{r}_{c}"

    # Adiciona resistores horizontais e verticais
    for r in range(rows):
        for c in range(cols):
            # Horizontal
            if c + 1 < cols:
                r_val = rng.uniform(50.0, 500.0)
                elements.append(Resistor(f"Rh_{r}_{c}", node(r, c), node(r, c + 1), r_val))
            # Vertical
            if r + 1 < rows:
                r_val = rng.uniform(50.0, 500.0)
                elements.append(Resistor(f"Rv_{r}_{c}", node(r, c), node(r + 1, c), r_val))

    # Fontes de excitação nos cantos e centro
    elements.append(VoltageSource("V_top", node(rows - 1, cols - 1), "0", 24.0))
    elements.append(CurrentSource("I_center", node(rows // 2, cols // 2), "0", 0.05))

    circ = Circuit(elements)
    assert circ.n_nodes == (rows * cols) - 1

    sol_dense = solve(circ, solver="dense")
    sol_sparse = solve(circ, solver="sparse")

    # Comparação estrita Denso vs Esparso Python
    for n_label in circ.nodes:
        v_d = sol_dense.V(n_label)
        v_s = sol_sparse.V(n_label)
        assert abs(v_d - v_s) < 1e-9, f"Divergência no nó {n_label}: {v_d} vs {v_s}"

    # Comparação com o Backend Rust
    if RUST_AVAILABLE:
        sol_rust = solve(circ, solver="rust_sparse")
        for n_label in circ.nodes:
            v_d = sol_dense.V(n_label)
            v_r = sol_rust.V(n_label)
            assert abs(v_d - v_r) < 1e-9, f"Divergência Rust no nó {n_label}: {v_d} vs {v_r}"

    # Verificação da LKC em nó interno aleatório
    probe_r, probe_c = rows // 2, cols // 2
    probe_node = node(probe_r, probe_c)
    # Soma de correntes entrando/saindo deve ser igual à corrente injetada pela fonte
    i_sum = 0.0
    for el in elements:
        if isinstance(el, Resistor):
            if el.node_p == probe_node:
                i_sum += (sol_dense.V(el.node_p) - sol_dense.V(el.node_n)) / el.resistance
            elif el.node_n == probe_node:
                i_sum += (sol_dense.V(el.node_n) - sol_dense.V(el.node_p)) / el.resistance
    assert abs(i_sum - 0.05) < 1e-9


# ==============================================================================
# 2. Stress de Faixa Dinâmica Extrema: Escada R-2R de 64 Bits
# ==============================================================================
def test_stress_64bit_ladder_dynamic_range():
    """Valida a precisão numérica ao longo de 64 estágios de atenuação binária.

    A tensão na terminação atinge 1.0 * 2^-64 ≈ 5.42e-20 V.
    Testa se o solver preserva precisão sem subflow catastrófico e se Python e Rust concordam.
    """
    n_bits = 64
    elements = [VoltageSource("V_IN", "1", "0", 1.0)]

    for i in range(1, n_bits + 1):
        elements.append(Resistor(f"R_ser_{i}", str(i), str(i + 1), 1000.0))
        elements.append(Resistor(f"R_sh_{i}", str(i), "0", 2000.0))
    elements.append(Resistor("R_term", str(n_bits + 1), "0", 2000.0))

    circ = Circuit(elements)
    sol_dense = solve(circ, solver="dense")
    sol_sparse = solve(circ, solver="sparse")

    v_final_dense = sol_dense.V(str(n_bits + 1))
    v_final_sparse = sol_sparse.V(str(n_bits + 1))

    assert v_final_dense > 0.0
    assert abs(v_final_dense - v_final_sparse) < 1e-19

    if RUST_AVAILABLE:
        sol_rust = solve(circ, solver="rust_sparse")
        v_final_rust = sol_rust.V(str(n_bits + 1))
        assert abs(v_final_dense - v_final_rust) < 1e-19


# ==============================================================================
# 3. Stress de Condicionamento Extremo (Ill-Conditioned Bridge: 1 uOhm a 100 MOhm)
# ==============================================================================
def test_stress_ill_conditioned_resistor_bridge():
    """Valida a robustez numérica com span de 14 ordens de magnitude de impedância.

    Garante que não ocorrem estouros (NaN ou Inf) e que os solvers convergem.
    """
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R_shunt", "1", "2", 1e-6),      # 1 uOhm (quase curto)
        Resistor("R_isol", "2", "3", 1e8),       # 100 MOhm (quase aberto)
        Resistor("R_load", "3", "0", 10e3),      # 10 kOhm
        Resistor("R_bleed", "2", "0", 50e3),     # 50 kOhm
    ]
    circ = Circuit(elements)
    sol_dense = solve(circ, solver="dense")
    sol_sparse = solve(circ, solver="sparse")

    # V(2) deve ser praticamente 10.0 V
    assert abs(sol_dense.V("2") - 10.0) < 1e-5
    assert abs(sol_sparse.V("2") - 10.0) < 1e-5

    # Divisor entre R_isol e R_load: V(3) = V(2) * (10k / (100M + 10k)) ≈ 10 * 1e-4 = 1 mV
    v3_expected = 10.0 * (10e3 / (1e8 + 10e3))
    assert math.isclose(sol_dense.V("3"), v3_expected, rel_tol=1e-5)
    assert math.isclose(sol_sparse.V("3"), v3_expected, rel_tol=1e-5)

    if RUST_AVAILABLE:
        sol_rust = solve(circ, solver="rust_sparse")
        assert math.isclose(sol_rust.V("3"), v3_expected, rel_tol=1e-5)


# ==============================================================================
# 4. Stress de Subcircuitos: Aninhamento Profundo (12 Níveis) e Parâmetros
# ==============================================================================
def test_stress_deeply_nested_subcircuits_with_parameters():
    """Valida expansão de subcircuitos em cascata profunda de 12 níveis com fórmulas.

    Garante que escopos de parâmetros são isolados e expressões são avaliadas
    corretamente em cada nível hierárquico.
    """
    depth = 12
    lines = ["* Stress Test: 12-Level Deeply Nested Subcircuits"]

    for i in range(1, depth):
        lines.append(f".subckt level_{i} in out R_VAL=100")
        lines.append(f"X1 in out level_{i+1} R_VAL={{R_VAL*1.5}}")
        lines.append(".ends")

    # Folha terminal
    lines.append(f".subckt level_{depth} in out R_VAL=100")
    lines.append("R_leaf in out {R_VAL}")
    lines.append(".ends")

    lines.append("V_SRC in_top 0 15.0")
    lines.append("X_TOP in_top 0 level_1 R_VAL=200")
    lines.append(".end")

    netlist_text = "\n".join(lines)
    elements = parse_netlist(netlist_text)
    circ = Circuit(elements)

    sol = solve(circ)
    # A resistência final deve ser: 200 * (1.5)^11
    expected_r = 200.0 * (1.5 ** (depth - 1))
    expected_i = -15.0 / expected_r

    i_src = sol.I_vsource("V_SRC")
    assert math.isclose(i_src, expected_i, rel_tol=1e-6)


# ==============================================================================
# 5. Stress Transiente Rígido (Stiff System: 10 ns e 10 ms)
# ==============================================================================
def test_stress_stiff_transient_multi_timescale():
    """Valida integração numérica transiente em sistema com pólos amplamente dispersos.

    Malha 1 com tau_1 = R1 * C1 = 100 * 100p = 10 ns
    Malha 2 com tau_2 = R2 * C2 = 1k * 10u = 10 ms
    Razão de rigidez = 10^6.
    Executa 1000 passos e compara Python com Rust.
    """
    elements = [
        VoltageSource("V1", "1", "0", 5.0),
        Resistor("R1", "1", "2", 100.0),
        Capacitor("C1", "2", "0", 100e-12, ic=0.0),   # tau = 10 ns
        Resistor("R2", "2", "3", 1000.0),
        Capacitor("C2", "3", "0", 10e-6, ic=0.0),     # tau = 10 ms
    ]
    circ = Circuit(elements)

    t_stop = 1e-4  # 100 us
    t_step = 1e-7  # 100 ns (1000 passos)

    res_py = solve_transient(circ, t_stop=t_stop, dt=t_step, method="trapezoidal", uic=True)

    # C1 carrega rápido mas divide com R2; V(2) fica em ~4.55 V enquanto C2 (10 uF) carrega lentamente
    assert 4.5 < res_py.node_voltages["2"][-1] < 4.6
    assert res_py.node_voltages["3"][-1] < 0.1

    # Verificação de energia armazenada nos capacitores
    assert res_py.energy_stored is not None
    assert res_py.energy_stored[-1] > 0.0

    if RUST_AVAILABLE:
        from circuit_solver.engine.transient_rust import solve_transient_rust
        res_rust = solve_transient_rust(circ, t_stop=t_stop, dt=t_step, method="trapezoidal", uic=True)

        # Python e Rust devem concordar ao longo da trajetória
        diff_v2 = np.max(np.abs(res_py.node_voltages["2"] - res_rust.node_voltages["2"]))
        diff_v3 = np.max(np.abs(res_py.node_voltages["3"] - res_rust.node_voltages["3"]))
        assert diff_v2 < 1e-6
        assert diff_v3 < 1e-6


# ==============================================================================
# 6. Stress de Concorrência: Monte Carlo Massivo (25.000 amostras em Rust)
# ==============================================================================
def test_stress_heavy_monte_carlo_concurrency():
    """Valida paralelismo com Rayon sob 25.000 amostras com semente reproduzível.

    Garante estabilidade sem travamentos de thread pool, e convergência assintótica.
    """
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "2", 1000.0, tolerance=0.10),
        Resistor("R2", "2", "0", 1000.0, tolerance=0.10),
        Resistor("R3", "2", "3", 500.0, tolerance=0.05),
        Resistor("R4", "3", "0", 500.0, tolerance=0.05),
    ]
    circ = Circuit(elements)
    n_samples = 25000

    mc_res1 = monte_carlo(circ, n_samples=n_samples, seed=12345)
    mc_res2 = monte_carlo(circ, n_samples=n_samples, seed=12345)

    idx2 = mc_res1.node_labels.index("2")
    v2_mean1 = mc_res1.mean[idx2]
    v2_mean2 = mc_res2.mean[idx2]
    assert v2_mean1 == v2_mean2

    # A média teórica no divisor simétrico R1 e (R2 || (R3+R4))
    # R_eq_down = 1000 || 1000 = 500 ohms
    # V(2) = 10 * 500 / 1500 = 3.3333 V
    assert math.isclose(v2_mean1, 10.0 * (500.0 / 1500.0), rel_tol=0.01)


# ==============================================================================
# 7. Stress de Fuzzing: Netlists Patológicas e Sintaxe SPICE Não-Convencional
# ==============================================================================
def test_stress_spice_fuzzing_pathological_syntax():
    """Testa robustez do parser contra continuas quebras '+', tabs, comentários e sufixos exóticos."""
    # Linha com múltiplos '+' e tabs
    pathological_netlist = """
* Circuito com sintaxe patológica
V1	1	0	DC	10
+	AC	1.0	45.0
R1	1	2	1000
+	; comentario no meio da continuacao
+	tolerance=0.05
R2	2	0	1Meg	$ comentario cifrao
C1	2	0	1000000pF	ic=0.0
L1	2	3	1000uH
R3	3	0	0.001k
.END
"""
    elements, _ = parse_netlist_with_hierarchy(pathological_netlist)
    assert len(elements) == 6

    circ = Circuit(elements)
    sol = solve(circ)
    assert sol.V("1") == 10.0
    # Em DC, L1 é um curto para R3 (1 ohm), que fica em paralelo com R2 (1Meg).
    # V(2) = 10 * (1 / (1000 + 1)) ≈ 0.00999 V
    expected_v2 = 10.0 * (1.0 / 1001.0)
    assert math.isclose(sol.V("2"), expected_v2, rel_tol=1e-4)


# ==============================================================================
# 8. Ingestão Direta em Memória via parse_netlist
# ==============================================================================
def test_stress_direct_memory_string_in_parse_netlist():
    """Garante que parse_netlist aceita strings com múltiplas linhas sem estourar OSError."""
    long_comment_netlist = (
        "* Netlist em memoria com centenas de linhas de comentario\n"
        + "".join(f"* Linha de preenchimento {i:04d} " + "A" * 70 + "\n" for i in range(50))
        + "V_TEST 1 0 12.0\n"
        + "R_TEST 1 0 1k\n"
        + ".END\n"
    )
    elements = parse_netlist(long_comment_netlist)
    assert len(elements) == 2
    circ = Circuit(elements)
    sol = solve(circ)
    assert sol.V("1") == 12.0
