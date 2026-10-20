# ⚡ pycircuit-solver

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)

**`pycircuit-solver`** é um motor computacional modular em Python para análise sistemática de circuitos elétricos lineares em corrente contínua (CC), fundamentado no método da **Análise Nodal Modificada (MNA — *Modified Nodal Analysis*)**.

O projeto foi arquitetado para atender rigorosamente ao escopo acadêmico dos **Capítulos 1 a 5** de *Fundamentos de Circuitos Elétricos* (Alexander & Sadiku), unindo precisão matemática estática, validação topológica robusta contra erros humanos e suporte a engenharia de software de nível de portfólio.

---

## 🎯 Escopo e Capacidades do Sistema

### 1. Cobertura Acadêmica (Referência: Alexander & Sadiku)
* **Capítulos 1 & 2:** Leis básicas, Lei de Ohm, LKC, LKT, associações série/paralelo, transformações $\Delta - Y$ e balanço de potência.
* **Capítulo 3:** Análise Nodal e de Malhas com todas as 4 fontes dependentes ($VCVS$, $VCCS$, $CCCS$, $CCVS$).
* **Capítulo 4:** Teoremas de Circuitos (Superposição, Thévenin, Norton e Máxima Transferência de Potência).
* **Capítulo 5:** Amplificadores Operacionais Ideais (circuitos inversores, não-inversores, somadores e diferenciais).
* **Capítulos 9 & 10:** Análise Fasorial Senoidal em Regime Permanente (CA), impedâncias/admitâncias complexas ($R, L, C$), potência complexa ($P, Q, S$, fator de potência) e resposta em frequência (Bode).

### 2. Funcionalidades Nativas
* **Motor MNA Dual (DC & CA):** Resolução matricial em números reais (DC) e complexos `complex128` (CA).
* **Fontes Dependentes Completas:** Suporte integral aos elementos SPICE $E$ (VCVS), $G$ (VCCS), $F$ (CCCS) e $H$ (CCVS).
* **Elementos Reativos (L e C):** Modelagem de capacitores e indutores em regime permanente senoidal.
* **CLI Integrada (`pycircuit`):** Resolução, Thévenin, Monte Carlo e análise CA diretamente no terminal.
* **Validador Topológico Pré-Solver (`validator.py`):** Checagem de nós flutuantes, terra obrigatório, fontes em conflito e fontes de controle válidas.
* **Conversor de Sufixos de Engenharia:** Leitura nativa de notações laboratoriais (`"4.7k"`, `"10mA"`, `"220u"`, `"4k7"`, `"1MEG"`).
* **Extrator de Thévenin/Norton:** Parâmetros $V_{\text{th}}$, $I_{\text{N}}$, $R_{\text{th}}$ e varredura $P_{\text{max}}$.
* **Auditoria de Potência (Tellegen):** Verificação de conservação global de energia ($\sum P = 0$) em DC e potência complexa em CA.
* **Simulação de Monte Carlo:** Análise estatística de tolerâncias ($1\%, 5\%, 10\%$).
* **Parser SPICE:** Leitura de arquivos `.cir`, `.net`, `.sp`.
* **Visualização:** Curvas de máxima potência, histogramas de Monte Carlo e diagramas de Bode.

---

## 🗂️ Arquitetura de Diretórios do Projeto

```text
pycircuit-solver/
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # Pipeline de testes automatizados com pytest
│
├── .gitignore
├── LICENSE                              # Licença MIT
├── README.md                            # Documentação técnica e guia de uso
├── pyproject.toml                       # Configuração do pacote Python moderno (PEP 621)
├── requirements.txt                     # Dependências mínimas de execução
│
├── circuit_solver/                      # Pacote principal
│   ├── __init__.py                      # Facade de importação pública
│   ├── cli.py                           # Interface de Linha de Comando (CLI pycircuit)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── elements.py                  # Dataclasses (R, C, L, V, I, VCVS, VCCS, CCCS, CCVS, OpAmp)
│   │   ├── circuit.py                   # Container de topologia e mapeamento de nós
│   │   ├── validator.py                 # Validador topológico e conversor de sufixos de engenharia
│   │   └── solution.py                  # Processamento de tensões, correntes e balanço de Tellegen
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── mna.py                       # Algoritmo MNA estático (DC)
│   │   ├── ac.py                        # Motor fasorial e resposta em frequência (CA)
│   │   ├── thevenin.py                  # Extração de Thévenin, Norton e Máxima Potência
│   │   └── monte_carlo.py               # Varredura estatística de tolerâncias
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   └── parser.py                    # Leitor de arquivos Netlist padrão SPICE (.cir)
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                     # Gráficos de potência, Monte Carlo e Diagramas de Bode
│
├── examples/                            # Scripts de problemas clássicos resolvidos
│   ├── cap02_ponte_wheatstone.py        # Ponte de Wheatstone desbalanceada + Monte Carlo
│   ├── cap04_thevenin_analise.py        # Equivalente Thévenin/Norton e curva P_max
│   ├── cap05_opamp_inversor.py          # Amplificador Operacional Inversor com terra virtual
│   └── cap09_filtro_rc_fasores.py       # Filtro passa-baixas RC, fasores e Diagrama de Bode
│
└── tests/                               # Suíte de validação analítica (pytest)
    ├── test_validator.py                # Testes de validação topológica e sufixos
    ├── test_mna_engine.py               # Testes analíticos do motor MNA e 4 fontes dependentes
    ├── test_thevenin.py                 # Testes de Thévenin/Norton e máxima transferência
    ├── test_ac_engine.py                # Testes de análise CA, ressonância e corte
    └── test_cli.py                      # Testes da interface de terminal pycircuit
```

---

## 💻 Interface de Linha de Comando (CLI)

O comando `pycircuit` fica disponível no terminal após a instalação:

```bash
# Resolver circuito DC a partir de netlist SPICE
pycircuit solve circuito.cir

# Extrair equivalente de Thévenin / Norton entre nós 3 e 0 com gráfico
pycircuit thevenin circuito.cir -a 3 -b 0 --plot

# Análise de Monte Carlo com 1000 amostras e tolerância de 5%
pycircuit monte-carlo circuito.cir -n 1000 -t 0.05 --plot

# Análise fasorial CA a 60 Hz
pycircuit ac circuito.cir -f 60.0

# Varredura de frequência e diagrama de Bode (10 Hz a 100 kHz)
pycircuit ac circuito.cir --sweep 10 100000 100 --node out --plot
```

---

## 🚀 Instalação e Uso Rápido

### Instalação

```bash
git clone https://github.com/elycbarros/pycircuit-solver.git
cd pycircuit-solver
pip install -e .
```

Para instalar dependências de desenvolvimento e testes:
```bash
pip install -e .[dev]
```

### Exemplo em Python (Divisor com Carga e Op-Amp)

```python
from circuit_solver import Circuit, VoltageSource, Resistor, OpAmp, solve

# Definindo os elementos do circuito
elements = [
    VoltageSource("Vin", "in", "0", 2.0),
    Resistor("R1", "in", "vn", 10_000.0),
    Resistor("Rf", "vn", "vout", 50_000.0),
    Resistor("Rload", "vout", "0", 10_000.0),
    OpAmp("U1", in_p="0", in_n="vn", out="vout", gnd="0")
]

circuit = Circuit(elements)
solution = solve(circuit)

print(solution.summary())
print(f"Ganho: {solution.V('vout') / solution.V('in'):.2f}")  # Av = -5.00
```

### Extração de Thévenin e Norton

```python
from circuit_solver import thevenin_norton

# Extrair equivalente visto entre terminais "3" e "0"
th = thevenin_norton(circuit, node_a="3", node_b="0")
print(th.summary())
```

### Leitura de Netlist SPICE

```python
from circuit_solver.io.parser import parse_netlist_string
from circuit_solver import Circuit, solve

netlist = """
* Circuito Divisor de Tensão
V1 1 0 DC 12
R1 1 2 4k7
R2 2 0 10k
.end
"""

circuit = Circuit(parse_netlist_string(netlist))
sol = solve(circuit)
print(f"V(2) = {sol.V('2'):.4f} V")
```

---

## 🧪 Executando os Testes Automatizados

```bash
pytest
```

Com relatório de cobertura de código:
```bash
pytest --cov=circuit_solver --cov-report=term-missing
```

---

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE).
