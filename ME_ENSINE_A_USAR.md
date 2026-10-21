# 🎓 Me Ensine a Usar o `pycircuit-solver`!
*O Guia Prático, Visual e Passo a Passo para Analisar Circuitos com Python e Rust*

---

## 🚀 1. O Que é Este Projeto em 1 Minuto?

O **`pycircuit-solver`** é como uma "calculadora inteligente de circuitos elétricos" para o seu terminal e seus scripts Python. 

Você informa como os componentes estão conectados entre si através de **nós** (pontos de conexão), e o motor calcula automaticamente:
* A **tensão** em cada ponto do circuito.
* A **corrente** e a **potência** que passa em cada componente.
* O circuito equivalente de **Thévenin e Norton**.
* A resposta em **frequência (CA/Bode)** para indutores e capacitores.
* O comportamento real com **tolerância de fábrica (Monte Carlo acelerado em Rust)**.

> 💡 **A Única Regra de Ouro da Topologia:**  
> Todo circuito elétrico precisa de uma referência de zero volts. No `pycircuit-solver`, o nó de referência (o **Terra / GND**) **sempre deve se chamar `"0"`**.

---

## 🛠️ 2. Instalação Expressa

No seu terminal (com Python 3.10+):

```bash
# 1. Clone o repositório
git clone https://github.com/elycbarros/pycircuit-solver.git
cd pycircuit-solver

# 2. Instale o pacote em modo de desenvolvimento
pip install -e .

# 3. (Opcional) Instale as dependências para gráficos e testes
pip install -e .[dev]
```

Pronto! Agora você tem tanto a biblioteca Python quanto o comando `pycircuit` disponível no seu terminal.

---

## 🧪 3. Lição 1: Seu Primeiro Circuito em Python

Vamos resolver um divisor de tensão clássico: uma fonte de $12\text{ V}$ ligada a dois resistores ($R_1 = 4\,\Omega$ e $R_2 = 6\,\Omega$).

```text
       Nó 1               Nó 2
 ( + )───────[ R1: 4Ω ]─────────┬─────── ( - )
   │                            │
[ V1: 12V ]                 [ R2: 6Ω ]
   │                            │
 ( - )──────────────────────────┴─────── Terra: "0"
```

Crie um arquivo `meu_circuito.py` ou abra o terminal Python (`python3`):

```python
from circuit_solver import Circuit, VoltageSource, Resistor, solve

# 1. Descreva os componentes e como eles se ligam nos nós
elementos = [
    VoltageSource("V1", node_p="1", node_n="0", voltage=12.0),
    Resistor("R1", node_p="1", node_n="2", resistance=4.0),
    Resistor("R2", node_p="2", node_n="0", resistance=6.0),
]

# 2. Crie o circuito
circuito = Circuit(elementos)

# 3. Resolva o circuito!
solucao = solve(circuito)

# 4. Imprima os resultados
print(solucao.summary())
```

### O que o terminal vai imprimir:
```text
========================================================================
  MNA Solution Summary
========================================================================
  Node Voltages:
    V(1) = 12.000000 V
    V(2) =  7.200000 V
------------------------------------------------------------------------
  Branch Currents:
    I(R1) =  1.200000 A
    I(R2) =  1.200000 A
    I(V1) = -1.200000 A
========================================================================
```

Você também pode acessar qualquer valor diretamente no seu código:
```python
v2 = solucao.V("2")         # 7.2 Volts
i_r1 = solucao.I("R1")      # 1.2 Amperes
p_r1 = solucao.P("R1")      # 5.76 Watts
```

---

## 🏷️ 4. Lição 2: Use Notações de Bancada (`"10k"`, `"4k7"`, `"100u"`)

Na vida real, ninguém escreve `4700` ou `0.000010`. O `pycircuit-solver` entende sufixos de engenharia nativamente:

```python
from circuit_solver import parse_value

r_base = parse_value("4k7")    # 4700.0 Ohms
c_filtro = parse_value("100u") # 0.0001 Farads
v_fonte = parse_value("12V")   # 12.0 Volts
r_mega = parse_value("1MEG")   # 1_000_000.0 Ohms
```

---

## ⚡ 5. Lição 3: Usando Arquivos de Netlist SPICE e a CLI

Não quer escrever código Python? Você pode descrever o circuito em um arquivo de texto comum (estilo SPICE `.cir`):

Crie o arquivo `divisor.cir`:
```spice
* Divisor de Tensão Simples
V1 1 0 12V
R1 1 2 4.7k
R2 2 0 10k
.end
```

Agora, basta rodar direto no terminal:
```bash
pycircuit solve divisor.cir
```

E o terminal exibirá a solução completa instantaneamente!

---

## 🔍 6. Lição 4: Equivalente de Thévenin e Norton

Quer simplificar uma rede elétrica inteira conectada entre dois terminais e descobrir onde ocorre a **máxima transferência de potência**?

```bash
# Extrai Thévenin entre o nó 2 e o terra "0", abrindo o gráfico da curva P_max
pycircuit thevenin divisor.cir -a 2 -b 0 --plot
```

Em Python:
```python
from circuit_solver.engine.thevenin import thevenin_norton

th = thevenin_norton(circuito, node_a="2", node_b="0")

print(f"Vth:  {th.v_th:.3f} V")
print(f"Rth:  {th.r_th:.3f} Ω")
print(f"In:   {th.i_norton:.3f} A")
print(f"Pmax: {th.p_max:.3f} W (quando R_load = {th.r_th:.3f} Ω)")
```

---

## 🎛️ 7. Lição 5: Circuitos com Amplificador Operacional (Op-Amp)

Vamos montar um **Amplificador Inversor** com ganho $A_v = -R_f / R_1 = -5$:

```text
               Rf: 50k
        ┌──────[===]──────┐
        │                 │
Vin ──[===]───┬───────────┴── Vout
     R1: 10k  │   |\
              └───|- \
                  |   >────── Vout
              ┌───|+ /
             _│_  |/
             /// (GND)
```

No Python:
```python
from circuit_solver import Circuit, VoltageSource, Resistor, OpAmp, solve

circuito_opamp = Circuit([
    VoltageSource("Vin", "in", "0", 2.0),
    Resistor("R1", "in", "vn", 10_000.0),
    Resistor("Rf", "vn", "vout", 50_000.0),
    Resistor("Rload", "vout", "0", 10_000.0),
    # in_p = "+", in_n = "-", out = "saída", gnd = "0"
    OpAmp("U1", in_p="0", in_n="vn", out="vout", gnd="0")
])

sol = solve(circuito_opamp)

print(f"Tensão de Entrada: {sol.V('in'):.2f} V")
print(f"Terra Virtual:     {sol.V('vn'):.6f} V  (idealmente 0V)")
print(f"Tensão de Saída:   {sol.V('vout'):.2f} V (Ganho exato de -5x)")
```

---

## 🌊 8. Lição 6: Análise em Corrente Alternada (CA Fasorial) e Filtros

Para circuitos com capacitores e indutores em regime permanente senoidal:

Crie o arquivo `filtro_passa_baixas.cir`:
```spice
* Filtro Passa-Baixas RC (fc ≈ 159 Hz)
Vin in 0 AC 1.0 0.0
R1 in out 1k
C1 out 0 1u
.end
```

### Análise em uma frequência específica (ex: 60 Hz):
```bash
pycircuit ac filtro_passa_baixas.cir -f 60.0
```

### Varredura de Frequência e Diagrama de Bode Automático:
```bash
# Varrer de 1 Hz a 100 kHz com 100 pontos logarítmicos e plotar Bode
pycircuit ac filtro_passa_baixas.cir --sweep 1 100000 100 --node out --plot
```

---

## ⏱️ 9. Lição 7: Análise Transitória no Tempo (Carga/Descarga RC, RL e RLC)

Quer saber **como a tensão sobe ou desce com o passar dos microssegundos** quando você liga a chave da fonte?

### No Terminal (CLI):
```bash
# Simular de 0 a 10 ms com passo de 10 μs e abrir o gráfico da onda
pycircuit tran filtro_passa_baixas.cir --tstop 0.01 --dt 1e-5 --node out --plot
```

### No Python (com fonte de degrau ou pulso):
```python
from circuit_solver import (
    Circuit,
    VoltageSource,
    Resistor,
    Capacitor,
    StepWaveform,
    solve_transient,
)

# Circuito RC com degrau de 0V para 10V
circuito_rc = Circuit([
    VoltageSource("V1", "1", "0", 10.0, waveform=StepWaveform(v1=0.0, v2=10.0, t_step=0.0)),
    Resistor("R1", "1", "2", 1000.0),
    Capacitor("C1", "2", "0", 1e-6, ic=0.0),
])

# Simula 5 milissegundos (5 constantes de tempo tau)
sol_tran = solve_transient(circuito_rc, t_stop=0.005, dt=1e-5)

print(sol_tran.summary())
print(f"Tensão em 1 ms (1 tau): {sol_tran.V('2')[100]:.3f} V  (≈ 63.2% de 10V)")
print(f"Tensão final em 5 ms:   {sol_tran.V('2')[-1]:.3f} V   (≈ 99.3% de 10V)")
```

---

## 🎲 10. Lição 8: Análise Estatística de Monte Carlo (Acelerada em Rust)

Resistores comerciais nunca têm o valor exato: eles têm tolerâncias de $1\%$, $5\%$ ou $10\%$.  
O que acontece com a tensão da sua placa quando $10.000$ circuitos forem fabricados?

No terminal:
```bash
# Roda 10.000 simulações com 5% de tolerância e plota o histograma com P5 e P95
pycircuit monte-carlo divisor.cir -n 10000 -t 0.05 --plot
```

No Python (usando o acelerador em Rust multi-thread):
```python
from circuit_solver.engine.monte_carlo_rust import monte_carlo

# Roda 50.000 simulações em poucos milissegundos
resultado = monte_carlo(circuito, n_samples=50_000, seed=42)

print(resultado.summary())
```

### O que você verá:
```text
========================================================================
  Monte Carlo Analysis  (50000 samples, 0 failed)
========================================================================
    Node      Mean (V)     Std (V)    Min (V)    Max (V)    P5 (V)    P95 (V)
  ----------------------------------------------------------------------
       1     12.000000    0.000000  12.000000  12.000000  12.000000  12.000000
       2      7.199412    0.165241   6.521840   7.842105   6.928190   7.471830
========================================================================
```
* **P5 e P95:** Em $90\%$ dos circuitos fabricados, a tensão no nó 2 ficará garantidamente entre $6.92\text{ V}$ e $7.47\text{ V}$.

---

## ⚡ 11. Lição: Semicondutores & Solver Não-Linear (Diodos, Zener, BJT e MOSFET)

Circuitos com semicondutores não obedecem à Lei de Ohm linear ($I = V/R$). O `pycircuit-solver` resolve esses circuitos automaticamente usando o método iterativo de **Newton-Raphson**:

```python
from circuit_solver import Circuit, VoltageSource, Resistor, Diode, ZenerDiode, BJT, MOSFET, solve

# 1. Diodo Retificador PN (Equação de Shockley)
circ_diodo = Circuit([
    VoltageSource("V1", "1", "0", 5.0),
    Resistor("R1", "1", "2", 1000.0),
    Diode("D1", "2", "0"),  # Ânodo no nó 2, Cátodo no nó 0
])
sol_d = solve(circ_diodo)
print(f"Diodo: V_D = {sol_d.V('2'):.4f} V, I_D = {sol_d.I_diode('D1')*1e3:.2f} mA")

# 2. Regulador de Tensão com Diodo Zener (5.1V)
circ_zener = Circuit([
    VoltageSource("VIN", "1", "0", 12.0),
    Resistor("RS", "1", "2", 1000.0),
    ZenerDiode("Z1", "0", "2", Vz=5.1, Rz=10.0),  # Ânodo no terra, Cátodo no nó 2
])
sol_z = solve(circ_zener)
print(f"Zener Regulado: V_out = {sol_z.V('2'):.4f} V")

# 3. Chave com Transistor BJT NPN saturado
circ_bjt = Circuit([
    VoltageSource("VCC", "cc", "0", 5.0),
    VoltageSource("VIN", "in", "0", 5.0),
    Resistor("RC", "cc", "c", 1000.0),
    Resistor("RB", "in", "b", 10000.0),
    BJT("Q1", node_c="c", node_b="b", node_e="0", polarity="NPN", beta=100.0),
])
sol_q = solve(circ_bjt)
print(f"BJT Saturado: V_CE = {sol_q.V('c'):.4f} V, I_C = {sol_q.I_bjt('Q1')['C']*1e3:.2f} mA")
```

---

## 📈 12. Lição: Varredura Paramétrica DC (.dc sweep)

Em vez de calcular o circuito para apenas um valor estático de fonte ou resistência, você pode **varrer uma faixa contínua de valores** para traçar curvas de transferência ($V_{out} \times V_{in}$), curvas de carga ou famílias de curvas características:

### No terminal (CLI):
```bash
# Varre a fonte V1 de 0V a 10V com passo de 0.1V e plota a tensão no nó 'out'
pycircuit dc circuito.cir --sweep V1 0 10 0.1 --node out --plot
```

### No script Python:
```python
from circuit_solver import Circuit, VoltageSource, Resistor, dc_sweep, plot_dc_sweep

# Divisor resistivo
circuito = Circuit([
    VoltageSource("VIN", "in", "0", 0.0),
    Resistor("R1", "in", "out", 1000.0),
    Resistor("R2", "out", "0", 1000.0),
])

# Varre VIN de 0V a 10V em passos de 0.5V
res = dc_sweep(circuito, param_name="VIN", start=0.0, stop=10.0, step=0.5)

print(res.summary())
print("Tensões de saída V(out):", res.V("out"))

# Plot interativo
plot_dc_sweep(res, node_to_plot="out", title="Curva de Transferência V_out x V_in", show=True)
```

---

## 🧲 13. Transformadores e Acoplamento Magnético Mútuo

Você pode modelar indutores acoplados magneticamente por fluxo compartilhado de duas maneiras:
1. Usando a classe `Transformer` (transformador de 2 enrolamentos com terminais primário e secundário).
2. Usando indutores `Inductor` separados acoplados pelo elemento `MutualCoupling` (elemento `K` do SPICE).

### Exemplo com `Transformer` (Relação de Transformação 2:1):
```python
from circuit_solver import Circuit, VoltageSource, Transformer, Resistor, solve_ac

# Transforma 120 V em 60 V (relação de espiras n = sqrt(L1 / L2) = sqrt(10 / 2.5) = 2.0)
circuito = Circuit([
    VoltageSource("V1", "1", "0", voltage=120.0, ac_mag=120.0),
    Transformer("T1", p1="1", n1="0", p2="2", n2="0", L1=10.0, L2=2.5, coupling=0.999),
    Resistor("RL", "2", "0", 100.0),
])

sol = solve_ac(circuito, frequency=60.0)
print(f"Tensão Primário: {sol.V_mag('1'):.2f} V")
print(f"Tensão Secundário: {sol.V_mag('2'):.2f} V")  # ~60.0 V
```

### Exemplo em Netlist SPICE com elemento `K`:
```spice
* Circuito com Bobinas Acopladas
V1 1 0 AC 12 0
R1 1 2 4
L1 2 0 8
L2 3 0 10
C1 3 4 0.5
R2 4 0 5
K1 L1 L2 0.1118
.end
```

---

---

## 🧩 14. Lição: Subcircuitos SPICE Hierárquicos (`.subckt` / `.ends`) e Introspecção White-Box

Em projetos modulares e no SPICE industrial, não escrevemos circuitos gigantes em um arquivo plano: dividimos o sistema em **subcircuitos reutilizáveis** (blocos como amplificadores, filtros, portas lógicas e pontes H).

### Exemplo em Netlist SPICE:
```spice
* Circuito principal com bloco divisor modular
.subckt DIVISOR in out gnd PARAMS: R1=10k R2=10k
R1 in out {R1}
R2 out gnd {R2}
.ends

V1 vin 0 DC 12
X1 vin vmid 0 DIVISOR PARAMS: R1=4k7 R2=10k
.end
```

### Usando no Terminal (CLI) com Visão Didática White-Box:
O `pycircuit` não é uma caixa preta! Ele explica detalhadamente como o subcircuito foi expandido:

```bash
# Relatório White-Box: mostra o mapeamento de pinos e o impacto na matriz MNA
pycircuit explain circuito_com_subckt.cir

# Desenha a árvore hierárquica ASCII dos blocos
pycircuit hierarchy circuito_com_subckt.cir
```

### Macromodelos Prontos de Livros-Texto:
O `pycircuit` já traz uma biblioteca embutida (`circuit_solver.core.builtin_library`):
* `OPAMP_741`: Macromodelo com ganho finito $A_{OL}=200.000$, $R_{in}=2\text{ M}\Omega$ e $R_{out}=75\ \Omega$.
* `CMOS_INVERTER`: Inversor digital CMOS complementar.
* `SALLEN_KEY_LP`: Filtro ativo passa-baixas de 2ª ordem.

---

## 🔬 15. Lição: Ponto de Operação DC (`.op`) e Pequenos Sinais em Semicondutores

Ao simular amplificadores com transistores e diodos, você precisa saber:
1. **Onde o componente está polarizado?** (Ponto de Operação Quiescente $Q$: $I_D, I_C, V_{BE}, V_{CE}, V_{GS}$).
2. **Qual é o modelo linear incremental?** ($r_d$ para diodo; $g_m, r_\pi, r_o$ para BJT; $g_m, r_{ds}$ para MOSFET).
3. **Qual é a resposta em frequência CA do amplificador?** (Diagrama de Bode com capacitor de acoplamento e ganho).

### No Terminal (CLI):
```bash
# Executa o cálculo .op e exibe a tabela didática completa do ponto Q
pycircuit op amplificador_ce.cir
```

### O que o terminal imprime:
```text
======================================================================
  PONTO DE OPERAÇÃO DC (.op) E PARÂMETROS DE PEQUENOS SINAIS
======================================================================
Componente: Q1 (BJT NPN)
  Região:          ATIVA DIRETA
  V_BE:            0.712 V
  V_CE:            5.240 V
  I_B:             24.31 uA
  I_C:             2.431 mA (Beta = 100.0)
  Transcondutância gm:  94.12 mS
  Resistência r_pi:     1.062 kΩ
  Resistência r_o:      43.28 kΩ (VA = 100V)
----------------------------------------------------------------------
```

---

## 🎯 16. Lição: Análise de Sensibilidade Analítica (Rede Adjunta), Pior Caso (Worst-Case / EVA) e RSS

Na bancada real, nenhum resistor de $10\text{ k}\Omega$ tem exatamente $10.000\ \Omega$; ele varia dentro da tolerância ($\pm 1\%, \pm 5\%, \pm 10\%$).

Como saber:
* **Qual componente tem maior impacto na tensão de saída?** (Sensibilidade Relativa Normalizada $S_p^{V}$ em %/%).
* **Qual é a tensão mínima e máxima absoluta possível?** (Worst-Case / EVA — *Extreme Value Analysis*).
* **Qual é a dispersão estatística realista a $3\sigma$?** (Fórmula RSS — *Root-Sum-Square*).

### No Terminal (CLI):
```bash
# Análise completa de pior caso no nó 'out'
pycircuit sensitivity circuito.cir --target out

# Ou com o alias rápido 'wca' com 5% de tolerância
pycircuit wca circuito.cir -t out --tol 0.05
```

### No Python:
```python
from circuit_solver import Circuit, VoltageSource, Resistor, analyze_sensitivity

circuito = Circuit([
    VoltageSource("V1", "1", "0", 10.0),
    Resistor("R1", "1", "2", 1000.0, tolerance=0.05),
    Resistor("R2", "2", "0", 1000.0, tolerance=0.05),
])

res = analyze_sensitivity(circuito, target_node="2")
print(res.summary())
```

### O que você verá no relatório de Pareto:
```text
======================================================================
  ANÁLISE DE SENSIBILIDADE E PIOR CASO (WCA / EVA) - Nó '2'
  Tensão Nominal: 5.0000 V
======================================================================
1. RANKING DE PARETO DE SENSIBILIDADE:
  Componente   dV/dp         Sensibilidade Normalizada (%/%)   Tolerância
  -----------------------------------------------------------------------
  R2           +0.00250 V/Ω  +0.5000 %/%                       ±5.0 %
  R1           -0.00250 V/Ω  -0.5000 %/%                       ±5.0 %
  V1           +0.50000 V/V  +1.0000 %/%                       ±0.0 %

2. EXCURSÃO DE PIOR CASO (WORST-CASE):
  * EVA Exato (Extreme Value):  4.7619 V a 5.2381 V (Δ = ±0.2381 V)
  * Taylor Linear (1ª Ordem):   4.7500 V a 5.2500 V (Δ = ±0.2500 V)
  * Dispersão Estatística RSS:  4.8232 V a 5.1768 V (Δ = ±0.1768 V)
======================================================================
```

---

## ⚡ 17. Lição: Motor Transiente em Rust de Ultra Alta Performance (>1.000.000 passos/s) com Auditoria Energética

Para simular milhares de passos temporais (ex: 50.000 pontos em circuitos osciladores ou pulsados), rodar o loop em Python puro pode levar alguns segundos.

O `pycircuit` inclui um **motor transiente nativo em Rust** (`rust_core`), integrado via PyO3:
1. **Fatoração LU Única:** Fatora a matriz MNA invariante uma única vez em $O(N^3)$.
2. **Resolução Ultra-Rápida:** Cada passo temporal é resolvido em $O(N^2)$ por substituição em código Rust compilado.
3. **Auditoria de Conservação de Energia:** Mede em tempo real:
   * Energia armazenada em capacitores e indutores: $E(t) = \frac{1}{2} C v^2 + \frac{1}{2} L i^2$.
   * Potência dissipada em calor por efeito Joule nos resistores: $P_{\text{diss}}(t) = v_R^2 / R$.

### No Terminal (CLI):
```bash
# Executa o transiente usando o backend em Rust com gráfico
pycircuit tran circuito_rlc.cir --tstop 10m --dt 10u --backend rust --plot
```

### No Python:
```python
from circuit_solver import Circuit, VoltageSource, Resistor, Capacitor, solve_transient

circuito = Circuit([
    VoltageSource("V1", "1", "0", 10.0),
    Resistor("R1", "1", "2", 1000.0),
    Capacitor("C1", "2", "0", 1e-6),
])

# Simula com backend 'rust' (ou 'auto' para detecção transparente)
res = solve_transient(circuito, t_stop=5e-3, dt=1e-5, backend="rust")
print(f"Passos resolvidos: {len(res.t)}")
print(f"Energia final armazenada no capacitor: {res.stored_energy[-1] * 1e6:.2f} µJ")
```

---

## 🌐 18. Lição: Redes de Duas Portas (Quadripolos) e Sistemas Trifásicos

### 18.1 Redes de Duas Portas (`two_port.py`)
Extraia diretamente as matrizes de Impedância ($Z$), Admitância ($Y$) e Transmissão ($ABCD$):

```python
from circuit_solver import Circuit, Resistor, analyze_two_port

# Rede T assimétrica: R1=20Ω, R2=40Ω, R3=30Ω (shunt)
circuito = Circuit([
    Resistor("R1", "n1", "mid", 20.0),
    Resistor("R2", "n2", "mid", 40.0),
    Resistor("R3", "mid", "0", 30.0),
], allowed_open_nodes=["n1", "n2"])

res = analyze_two_port(circuito, port1=("n1", "0"), port2=("n2", "0"))
print(f"Z11 = {res.z11} Ω, Z22 = {res.z22} Ω")
print(f"Matriz ABCD: A={res.A:.3f}, B={res.B:.3f} Ω, C={res.C:.4f} S, D={res.D:.3f}")
print(f"Recíproco: {res.is_reciprocal}, Simétrico: {res.is_symmetric}")

# Impedância de entrada terminada em carga RL = 50 Ω
zin = res.input_impedance(Z_load=50.0)
print(f"Zin(RL=50) = {zin.real:.3f} Ω")
```

### 18.2 Sistemas Trifásicos e Método dos Dois Wattímetros (`three_phase.py`)
Gere geradores balanceados, cargas em triângulo ou estrela e meça a potência total via Teorema de Blondel:

```python
from circuit_solver import (
    Circuit, build_three_phase_source, build_delta_load,
    solve_ac, analyze_two_wattmeters
)

# Fonte equilibrada Y (120 V rms, 60 Hz) + Carga Delta (24 + j18 Ω)
fontes = build_three_phase_source("Y", V_rms=120.0, node_a="s_a", node_b="s_b", node_c="s_c", node_neutral="0")
cargas = build_delta_load(R=24.0, L=18.0/(2*3.14159*60), node_a="s_a", node_b="s_b", node_c="s_c")
circuito = Circuit(fontes + cargas)

sol = solve_ac(circuito, frequency=60.0)
z_branch = 24.0 + 18.0j
i_ab = (sol.V("s_a") - sol.V("s_b")) / z_branch
i_bc = (sol.V("s_b") - sol.V("s_c")) / z_branch
i_ca = (sol.V("s_c") - sol.V("s_a")) / z_branch
i_line_a = i_ab - i_ca
i_line_b = i_bc - i_ab

wm = analyze_two_wattmeters(
    V_line_ac=sol.V("s_a") - sol.V("s_c"),
    V_line_bc=sol.V("s_b") - sol.V("s_c"),
    I_line_a=i_line_a,
    I_line_b=i_line_b,
)
print(f"P_total = {wm.P_total:.1f} W, Q_total = {wm.Q_total:.1f} VAR, FP = {wm.power_factor:.3f}")
```

---

## 📋 19. Tabela de Cola dos Elementos Suportados

| Elemento | Letra SPICE | Como Instanciar em Python | O que faz |
| :--- | :---: | :--- | :--- |
| **Resistor** | `R` | `Resistor("R1", "p", "n", 100.0, tolerance=0.05)` | Resistor linear em Ohms |
| **Capacitor** | `C` | `Capacitor("C1", "p", "n", 10e-6, ic=0.0)` | Capacitor ideal em Farads |
| **Indutor** | `L` | `Inductor("L1", "p", "n", 1e-3, ic=0.0)` | Indutor ideal em Henries |
| **Acoplamento Mútuo** | `K` | `MutualCoupling("K1", "L1", "L2", coupling=0.999)` | Acoplamento magnético mútuo SPICE |
| **Transformador** | `T` | `Transformer("T1", "p1", "n1", "p2", "n2", L1=10.0, L2=2.5, coupling=0.999)` | Transformador de dois enrolamentos |
| **Fonte de Tensão** | `V` | `VoltageSource("V1", "p", "n", 12.0, ac_mag=1.0)` | Fonte DC, CA ou Transitória |
| **Fonte de Corrente** | `I` | `CurrentSource("I1", "p", "n", 2.0)` | Fonte DC, CA ou Transitória |
| **Diodo PN** | `D` | `Diode("D1", node_p="2", node_n="0", Is=1e-14, n=1.0)` | Diodo Shockley não-linear |
| **Diodo Zener** | `Z` | `ZenerDiode("Z1", node_p="0", node_n="2", Vz=5.1, Rz=10.0)` | Diodo com regulação por breakdown |
| **Transistor BJT** | `Q` | `BJT("Q1", "c", "b", "e", polarity="NPN", beta=100.0)` | Transistor Bipolar (Ebers-Moll) |
| **Transistor MOSFET**| `M` | `MOSFET("M1", "d", "g", "s", polarity="NMOS", Vth=1.0, Kp=1e-3)` | MOSFET canal N ou P |
| **Subcircuito SPICE**| `X` | `.subckt NOME p1 p2 ... .ends` / instanciado via `X1 ...` | Bloco hierárquico modular reutilizável |
| **VCVS** | `E` | `VCVS("E1", "p", "n", "cp", "cn", gain=2.0)` | Tensão controlada por tensão |
| **VCCS** | `G` | `VCCS("G1", "p", "n", "cp", "cn", transconductance=0.01)` | Corrente controlada por tensão |
| **CCCS** | `F` | `CCCS("F1", "p", "n", vctrl_name="V1", gain=5.0)` | Corrente controlada por corrente |
| **CCVS** | `H` | `CCVS("H1", "p", "n", vctrl_name="V1", transresistance=100.0)` | Tensão controlada por corrente |
| **Op-Amp** | `U` ou `A` | `OpAmp("U1", in_p="0", in_n="vn", out="vout", gnd="0")` | Amplificador Operacional ideal |

> 💡 **Nota sobre Fontes de Corrente em SPICE (`I` e `G`):**  
> Na convenção padrão SPICE (`I1 n+ n- 2.0`), a corrente sai do terminal positivo `n+`, atravessa a fonte e entra no terminal `n-` (ou seja, ela é injetada no nó `n-` do resto do circuito). O parser do `pycircuit` traduz isso automaticamente mapeando `n-` para o polo `node_p` do objeto `CurrentSource`.

---

## ⚠️ 20. Erros Comuns e Como Evitá-los

1. **"Missing ground node '0'":**
   * *Motivo:* Nenhum nó do circuito se chama `"0"`.
   * *Solução:* Renomeie o nó negativo da fonte ou o terra para `"0"`.

2. **"Circuit has floating sub-circuit":**
   * *Motivo:* Um grupo de componentes não tem nenhum caminho condutivo até o terra `"0"`.
   * *Solução:* Conecte uma referência ao terra nesse bloco de circuito.

3. **"Loop of ideal voltage sources":**
   * *Motivo:* Duas fontes de tensão ideais foram ligadas em paralelo direto, ou em um anel fechado violando a Lei das Malhas de Kirchhoff.
   * *Solução:* Adicione a resistência interna das fontes (ex: $0.1\,\Omega$) em série.

4. **"Tolerance must be positive":**
   * *Motivo:* Ao definir tolerância em resistores ou na CLI (`--tol`), foi passado um valor negativo ou nulo.
   * *Solução:* Use valores positivos (ex: `0.05` para 5%).

---

Agora você está pronto para simular qualquer circuito com precisão acadêmica e performance profissional! 🚀


