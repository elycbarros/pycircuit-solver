# 📘 Documentação Técnica Completa: `pycircuit-solver`
*Arquitetura de Software, Modelagem Matemática e Engenharia de Circuitos Elétricos*

---

## 📑 Sumário Executivo

O **`pycircuit-solver`** é uma plataforma computacional híbrida (Python + Rust) voltada para a análise, simulação e validação estática de circuitos elétricos lineares. O projeto foi projetado com duplo objetivo:
1. **Rigor Eletrotécnico:** Implementar os fundamentos clássicos da Teoria de Circuitos baseando-se no método da **Análise Nodal Modificada (MNA — *Modified Nodal Analysis*)**, com suporte integral a elementos passivos, ativos, fontes dependentes, regime permanente senoidal (CA fasorial), Teorema de Tellegen e teoremas de Thévenin/Norton.
2. **Engenharia de Software de Alto Nível:** Apresentar uma arquitetura desacoplada, tipagem estática rigorosa, validação topológica pré-solver baseada em Teoria dos Grafos, suíte automatizada de **447 testes** e um **núcleo de alta performance em Rust (`rust_core`)** que acelera simulações estatísticas (Monte Carlo) e transientes via multi-threading nativo sem bloqueio de GIL.

---

## 1. 📐 Fundamentação Teórica e Formulação Matemática

### 1.1 O Método da Análise Nodal Modificada (MNA)
A Análise Nodal convencional falha diretamente na presença de **fontes ideais de tensão**, pois a corrente que as atravessa é determinada pelo circuito externo e não por uma relação de condutância simples ($i = G \cdot v$). O método MNA (desenvolvido originalmente por Ho, Ruehli e Brennan em 1975) resolve essa limitação introduzindo a corrente das fontes de tensão como **variáveis auxiliares de estado**.

O sistema global de equações é expresso na forma matricial particionada em 4 blocos:

$$\begin{bmatrix} \mathbf{G} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix} \begin{bmatrix} \mathbf{v} \\ \mathbf{j} \end{bmatrix} = \begin{bmatrix} \mathbf{i} \\ \mathbf{e} \end{bmatrix}$$

Onde:
* $\mathbf{G} \in \mathbb{R}^{N \times N}$ (ou $\mathbb{C}^{N \times N}$ em CA): Submatriz de condutâncias nodais entre os $N$ nós não-terra do circuito.
* $\mathbf{B} \in \mathbb{R}^{N \times M}$: Matriz de incidência nó-ramo para os $M$ elementos que impõem restrição de tensão (fontes de tensão, VCVS, saídas de Op-Amps).
* $\mathbf{C} \in \mathbb{R}^{M \times N}$: Matriz de restrições de tensão impostas pelas fontes ou curto-circuito virtual.
* $\mathbf{D} \in \mathbb{R}^{M \times M}$: Matriz de acoplamento entre variáveis dependentes (nula para fontes ideais independentes; não-nula para CCVS ou modelos reais com resistência interna).
* $\mathbf{v} \in \mathbb{R}^{N}$: Vetor das tensões nodais incógnitas (medidas em relação ao terra $\text{node } 0 = 0\text{ V}$).
* $\mathbf{j} \in \mathbb{R}^{M}$: Vetor das correntes auxiliares incógnitas através dos ramos que impõem tensão.
* $\mathbf{i} \in \mathbb{R}^{N}$: Vetor de injeção nodal de corrente proveniente de fontes de corrente independentes.
* $\mathbf{e} \in \mathbb{R}^{M}$: Vetor de valores de restrição de tensão das fontes independentes.

---

### 1.2 Regras de Estampa Nodal (*Element Stamping*)

Cada elemento adicionado ao circuito injeta coeficientes aditivos diretamente nas posições correspondentes de $\mathbf{A} = \begin{bmatrix} \mathbf{G} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix}$ e $\mathbf{z} = \begin{bmatrix} \mathbf{i} \\ \mathbf{e} \end{bmatrix}$:

#### A. Resistor linear ($R$, condutância $g = 1/R$) entre nós $p$ e $n$:
A Lei de Ohm $i = g(v_p - v_n)$ resulta em contribuições na Lei de Kirchhoff das Correntes (KCL):
* Nó $p$: $\mathbf{G}[p, p] \mathrel{+}= g$, $\mathbf{G}[p, n] \mathrel{-}= g$
* Nó $n$: $\mathbf{G}[n, p] \mathrel{-}= g$, $\mathbf{G}[n, n] \mathrel{+}= g$
*(Se $p$ ou $n$ for o nó terra `'0'`, a respectiva linha/coluna é omitida pois $v_0 \equiv 0$).*

#### B. Fonte Independente de Tensão ($V$) entre nós $p$ e $n$, índice auxiliar $k$:
Impõe a equação $v_p - v_n = V$:
* Coluna $\mathbf{B}$: $\mathbf{A}[p, k] \mathrel{+}= 1$, $\mathbf{A}[n, k] \mathrel{-}= 1$ (injeção da corrente $j_k$ nos nós $p$ e $n$)
* Linha $\mathbf{C}$: $\mathbf{A}[k, p] \mathrel{+}= 1$, $\mathbf{A}[k, n] \mathrel{-}= 1$
* Vetor $\mathbf{e}$: $\mathbf{z}[k] = V$

#### C. Fonte Independente de Corrente ($I$) saindo de $n$ e entrando em $p$:
A corrente fixa entra no nó $p$ e sai de $n$:
* $\mathbf{z}[p] \mathrel{+}= I$
* $\mathbf{z}[n] \mathrel{-}= I$

> 💡 **Convenção SPICE:** Em netlists SPICE no formato `I1 n+ n- [DC val]`, a convenção padrão estipula que a corrente flui do terminal positivo `n+` para o negativo `n-` através da fonte (ou seja, ela entra no nó `n-` do circuito). O parser traduz isso automaticamente mapeando `n-` para o `node_p` do elemento `CurrentSource`, onde o solver estampa a injeção $+I$. O mesmo critério estrito de sentido aplica-se à fonte controlada VCCS ($G$). Se nenhum valor DC explícito for informado para fontes variante no tempo (`PULSE` ou `SIN`), o ponto de operação estático DC adota o valor da forma de onda em $t=0$ ($v_1$ para PULSE, $v_o$ para SIN).

#### D. Fontes Dependentes / Controladas (Elementos SPICE E, G, F, H):
1. **VCVS ($E$, ganho $\mu$):** Tensão de saída controlada por tensão $v_p - v_n = \mu (v_{cp} - v_{cn})$.
   * Coluna $B$: estampa usual de fonte de tensão no índice auxiliar $k$.
   * Linha de restrição $k$: $\mathbf{A}[k, p] \mathrel{+}= 1$, $\mathbf{A}[k, n] \mathrel{-}= 1$, $\mathbf{A}[k, cp] \mathrel{-}= \mu$, $\mathbf{A}[k, cn] \mathrel{+}= \mu$.
2. **VCCS ($G$, transcondutância $g_m$):** Corrente de saída controlada por tensão $i = g_m (v_{cp} - v_{cn})$.
   * Não requer variável auxiliar. Estampa direta no bloco $\mathbf{G}$:
   * $\mathbf{G}[p, cp] \mathrel{-}= g_m$, $\mathbf{G}[p, cn] \mathrel{+}= g_m$
   * $\mathbf{G}[n, cp] \mathrel{+}= g_m$, $\mathbf{G}[n, cn] \mathrel{-}= g_m$
3. **CCCS ($F$, ganho $\beta$):** Corrente controlada pela corrente que atravessa a fonte de controle $V_{ctrl}$.
   * Como a corrente $I(V_{ctrl}) = -j_{k_{ctrl}}$ (convenção gerador), a injeção na KCL do nó $p$ e $n$ é estampada na coluna auxiliar $k_{ctrl}$:
   * $\mathbf{A}[p, k_{ctrl}] \mathrel{+}= \beta$, $\mathbf{A}[n, k_{ctrl}] \mathrel{-}= \beta$.
4. **CCVS ($H$, transresistência $r_m$):** Tensão de saída controlada por corrente $v_p - v_n = r_m I(V_{ctrl})$.
   * Linha auxiliar de restrição $k$: $\mathbf{A}[k, p] \mathrel{+}= 1$, $\mathbf{A}[k, n] \mathrel{-}= 1$, $\mathbf{A}[k, k_{ctrl}] \mathrel{+}= r_m$.

#### E. Amplificador Operacional Ideal (Op-Amp):
Modelado em malha fechada ideal ($A_{OL} \to \infty, R_{in} \to \infty, R_{out} = 0$):
* **Curto-Circuito Virtual:** $v(in_p) - v(in_n) = 0 \implies \mathbf{A}[k, in_p] \mathrel{+}= 1, \mathbf{A}[k, in_n] \mathrel{-}= 1$.
* **Corrente de Saída:** A corrente necessária para manter a saída é fornecida pela variável auxiliar $k$: $\mathbf{A}[out, k] \mathrel{+}= 1$.

#### F. Indutores e Capacitores em Regime Permanente CC:
* **Capacitor em CC:** Comporta-se como um circuito aberto ideal ($i_C = C \frac{dv}{dt} = 0$). Não adiciona condutância nem ramo ao MNA.
* **Indutor em CC:** Comporta-se como um curto-circuito ideal ($v_L = L \frac{di}{dt} = 0$). É modelado via condutância de alta condutividade ($g = 10^6\ \text{S}$, equivalente a resistência parasita de $1\ \mu\Omega$).

---

### 1.3 Regime Permanente Senoidal (Análise CA Fasorial)
Para sinais puramente senoidais em frequência angular $\omega = 2\pi f$, o circuito é transformado no domínio da frequência utilizando fasores complexos:
* Indutor: Impedância $Z_L = j\omega L \implies Y_L = \frac{1}{j\omega L} = -j\frac{1}{\omega L}$.
* Capacitor: Impedância $Z_C = \frac{1}{j\omega C} \implies Y_C = j\omega C$.
* Resistor: Impedância $Z_R = R \implies Y_R = \frac{1}{R}$.

O sistema MNA torna-se complexo ($\mathbf{A}_{AC} \in \mathbb{C}^{M \times M}, \mathbf{z}_{AC} \in \mathbb{C}^{M}$), resolvido por decomposição LU em dupla precisão complexa (`complex128`).
As grandezas de saída são decompostas em:
$$V(t) = |V| \cos(\omega t + \theta) \iff \mathbf{V} = |V| e^{j\theta} = V_{real} + j V_{imag}$$

---

### 1.4 Teorema de Tellegen e Auditoria Energética
Em qualquer rede concentrada e linear, a conservação global de potência estipula que a soma algébrica das potências absorvidas e fornecidas em todos os ramos $B$ é rigorosamente nula:

$$\sum_{k=1}^{B} v_k \cdot i_k = 0$$

Em regime CA, a potência aparente complexa $S = P + jQ$ obedece ao mesmo princípio conservativo:

$$\sum_{k=1}^{B} \mathbf{V}_k \cdot \mathbf{I}_k^* = 0 \implies \sum P_k = 0 \quad \text{e} \quad \sum Q_k = 0$$

O `pycircuit-solver` audita automaticamente a conservação de energia através do método `Solution.power_audit()` e `ACSolution.power_audit()`, reportando desvios numéricos com tolerância inferior a $10^{-9}\text{ W}$.

---

### 1.5 Extração Dual de Thévenin e Norton
O cálculo analítico e computacional de circuitos equivalentes entre terminais arbitrários $A$ e $B$ opera através de duas etapas:
1. **Tensão de Circuito Aberto ($V_{th}$):** Resolve o circuito nominal sem carga entre $A$ e $B$: $V_{th} = V(A) - V(B)$.
2. **Resistência Equivalente ($R_{th}$):** Utiliza um **algoritmo de teste dual**:
   * **Injeção de Tensão:** Desativa fontes independentes e conecta uma fonte de teste $V_{test} = 1.0\text{ V}$ entre $A$ e $B$. A corrente $I_{test}$ é obtida da solução MNA, resultando em $R_{th} = 1.0 / I_{test}$.
   * **Injeção de Corrente (Fallback para $R_{th} = 0$):** Se houver uma fonte de tensão ou curto direto entre $A$ e $B$, a matriz MNA com fonte de teste de tensão torna-se singular. O motor detecta essa condição e comuta automaticamente para a **injeção de corrente** ($I_{test} = 1.0\text{ A}$), medindo a tensão resultante $\Delta V$:
     $$R_{th} = \frac{\Delta V}{I_{test}} = 0.0\ \Omega$$
3. **Corrente de Norton ($I_N$):** Calculada por $I_N = V_{th} / R_{th}$ quando $R_{th} > 0$.
4. **Máxima Transferência de Potência:** Ocorre quando $R_L = R_{th}$, fornecendo $P_{max} = \frac{V_{th}^2}{4 R_{th}}$.

---

### 1.6 Análise Transitória no Domínio do Tempo (`engine/transient.py`)
Para simular a dinâmica contínua $v(t)$ e $i(t)$ no tempo, as equações constitutivas diferenciais dos elementos reativos são discretizadas numericamente utilizando **modelos companheiros Norton (*companion models*)**:

#### A. Discretização do Capacitor ($i = C \frac{dv}{dt}$):
* **Regra dos Trapézios (*Trapezoidal Rule*, 2ª ordem, $A$-estável):**
  $$i_{k} = G_{eq} \cdot v_{k} - I_{eq, k-1}$$
  Onde:
  $$G_{eq} = \frac{2C}{\Delta t}, \quad I_{eq, k-1} = G_{eq} \cdot v_{k-1} + i_{k-1}$$
* **Euler Implícito (*Backward Euler*, 1ª ordem, $L$-estável):**
  $$G_{eq} = \frac{C}{\Delta t}, \quad I_{eq, k-1} = G_{eq} \cdot v_{k-1}$$

#### B. Discretização do Indutor ($v = L \frac{di}{dt}$):
* **Regra dos Trapézios:**
  $$i_k = G_{eq} \cdot v_k + I_{eq, k-1}$$
  Onde:
  $$G_{eq} = \frac{\Delta t}{2L}, \quad I_{eq, k-1} = G_{eq} \cdot v_{k-1} + i_{k-1}$$
* **Euler Implícito:**
  $$G_{eq} = \frac{\Delta t}{L}, \quad I_{eq, k-1} = i_{k-1}$$

#### C. Otimização por Fatoração LU Pré-Computada:
Para um passo de tempo fixo $\Delta t$, as condutâncias equivalentes $G_{eq}$ são estáticas ao longo de toda a simulação. Portanto, a matriz de coeficientes $\mathbf{A} \in \mathbb{R}^{M \times M}$ é invariante no tempo.
O solver realiza a **decomposição LU de $\mathbf{A}$ apenas uma vez antes do laço temporal** (`scipy.linalg.lu_factor`). A cada instante $t_k$, apenas o vetor $\mathbf{z}(t_k)$ é atualizado com as correntes históricas e fontes variáveis, resolvendo cada passo de tempo via substituição direta e reversa (`lu_solve`) em complexidade $O(M^2)$, permitindo $10.000$ passos temporais em frações de segundo.

---

### 1.7 Semicondutores & Solver de Newton-Raphson (`engine/nonlinear.py`)

Circuitos reais contêm dispositivos semicondutores com características corrente-tensão estritamente não-lineares. O `pycircuit-solver` implementa um motor de **Newton-Raphson** para linearizar iterativamente cada componente não-linear em torno do seu ponto de operação ($Q$-point) a cada iteração:

$$\mathbf{f}(\mathbf{x}) = \mathbf{A}_{lin} \mathbf{x} + \mathbf{i}_{nl}(\mathbf{x}) - \mathbf{z}_{lin} = \mathbf{0}$$

A linearização de primeira ordem via série de Taylor produz os modelos companheiros Norton:

$$\mathbf{i}_{nl}(\mathbf{x}) \approx \mathbf{i}_{nl}(\mathbf{x}^{(k)}) + \mathbf{J}^{(k)} (\mathbf{x} - \mathbf{x}^{(k)}) = \mathbf{J}^{(k)} \mathbf{x} - \mathbf{I}_{eq}^{(k)}$$

Onde a matriz Jacobiana $\mathbf{J}^{(k)} = \left.\frac{\partial \mathbf{i}_{nl}}{\partial \mathbf{x}}\right|_{\mathbf{x}^{(k)}}$ contém as condutâncias dinâmicas e transcondutâncias, e $\mathbf{I}_{eq}^{(k)} = \mathbf{J}^{(k)} \mathbf{x}^{(k)} - \mathbf{i}_{nl}(\mathbf{x}^{(k)})$ é o vetor de correntes equivalentes independentes estampadas no vetor $\mathbf{z}$.

A cada iteração $k$, o sistema linear resultante é resolvido:

$$(\mathbf{A}_{lin} + \mathbf{J}^{(k)}) \mathbf{x}^{(k+1)} = \mathbf{z}_{lin} + \mathbf{I}_{eq}^{(k)}$$

O algoritmo itera até que a norma do resíduo satisfaça o critério estrito de parada:

$$\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|_\infty < \epsilon \quad (\text{com } \epsilon = 10^{-6}\text{ V})$$

#### Modelos de Dispositivos Implementados:

1. **Diodo de Junção PN (Equação de Shockley):**
   $$I_D = I_S \left(e^{V_D / (n V_T)} - 1\right), \quad g_d = \frac{\partial I_D}{\partial V_D} = \frac{I_S}{n V_T} e^{V_D / (n V_T)}$$
   * Amortecimento exponencial SPICE (`pnjlim`): previne *overflow* numérico em ponto flutuante ($e^{80} \approx 5.5 \times 10^{34}$) limitando o avanço de tensão por passo em junções fortemente polarizadas.

2. **Diodo Zener:**
   * Polarização direta: modelo padrão de diodo de junção PN.
   * Regulação reversa (efeito Zener / avalanche): conduz bruscamente quando $V_D \le -V_Z$ com condutância dinâmica $g_Z = 1 / R_Z$:
     $$I_D = -I_S - \frac{(-V_Z) - V_D}{R_Z}$$

3. **Transistor Bipolar de Junção (BJT — NPN e PNP):**
   * Modelo acoplado de Ebers-Moll com diodos de base-emissor ($I_F$) e base-coletor ($I_R$):
     $$I_B = \frac{I_F}{\beta} + I_R, \quad I_C = I_F - 2 I_R, \quad I_E = -(I_B + I_C)$$
   * Modela naturalmente a região ativa ($V_{CE} > 0.2\text{ V}, I_C = \beta I_B$) e a saturação profunda ($V_{CE} \approx V_{CE,sat} \approx 0.05\text{ a }0.1\text{ V}$).

4. **Transistor de Efeito de Campo (MOSFET — NMOS e PMOS):**
   * Modelo quadrático clássico para as 3 regiões de operação:
     * **Corte ($V_{GS} \le V_{th}$):** $I_D = 0, g_m = 0, g_{ds} = g_{min} = 10^{-12}\ \Omega^{-1}$.
     * **Triodo / Linear ($V_{DS} < V_{GS} - V_{th}$):** Canal resistivo bidirecional:
       $$I_D = K_p \left[ (V_{GS} - V_{th}) V_{DS} - \frac{1}{2} V_{DS}^2 \right] (1 + \lambda V_{DS})$$
     * **Saturação ($V_{DS} \ge V_{GS} - V_{th}$):**
       $$I_D = \frac{1}{2} K_p (V_{GS} - V_{th})^2 (1 + \lambda V_{DS})$$
   * Amortecimento de canal FET (`fetlim`): impede oscilações estáticas de tensão entre corte e saturação.

---

### 1.8 Varredura Paramétrica DC (.dc sweep — `engine/dc_sweep.py`)

A análise de ponto de operação estático (.op) calcula as grandezas elétricas para um único conjunto de valores constantes. A **Varredura Paramétrica DC (.dc sweep)** estende essa capacidade, permitindo analisar o comportamento do circuito sob variação contínua de um parâmetro (tensão, corrente ou resistência):

1. **Curvas de Transferência ($V_{out} \times V_{in}$):**
   Mapeia o ganho de amplificadores, limiares de comutação de comparadores e inversores lógicos.
2. **Curvas de Carga ($V_L \times I_L$ ou $P_L \times R_L$):**
   Avalia a regulação de fontes e máxima transferência de potência sobre resistores de carga variáveis.
3. **Varredura Aninhada 2D (Família de Curvas Características):**
   Permite variar um parâmetro primário (ex: $V_{CE}$ de 0 a 5V) para múltiplos patamares discretos de um parâmetro secundário (ex: $I_B \in [10, 20, 30, 40, 50]\ \mu\text{A}$), gerando matrizes de resultados $\mathbf{V} \in \mathbb{R}^{N_2 \times N_1}$ e $\mathbf{I} \in \mathbb{R}^{N_2 \times N_1}$ e traçando diretamente as curvas características de dispositivos semicondutores ($I_C \times V_{CE}$, $I_D \times V_{DS}$).
4. **Resolução por ponto:**
   Cada ponto não linear é resolvido pelo Newton-Raphson; a varredura atual não reutiliza a solução anterior. Os pontos são start+k·step até o limite, sem alterar o circuito original.

---

### 1.9 Transformadores e Acoplamento Magnético Mútuo (Elemento K do SPICE — Alexander & Sadiku Cap. 13)

Circuito com indutores acoplados magneticamente por fluxo compartilhado (transformadores de dois enrolamentos e elemento de acoplamento mútuo SPICE `K`):

1. **Equações Constitutivas e Indutância Mútua:**
   A indutância mútua $M$ entre duas bobinas $L_1$ e $L_2$ com coeficiente de acoplamento magnético $k$ ($-1.0 \le k \le 1.0$) é regida por:
   $$M = k \sqrt{L_1 L_2}$$
   No domínio do tempo:
   $$\begin{bmatrix} v_1(t) \\ v_2(t) \end{bmatrix} = \begin{bmatrix} L_1 & M \\ M & L_2 \end{bmatrix} \frac{d}{dt}\begin{bmatrix} i_1(t) \\ i_2(t) \end{bmatrix} = \mathbf{L} \frac{d\mathbf{i}}{dt}$$
   A matriz de indutância inversa $\mathbf{\Gamma} = \mathbf{L}^{-1}$ (unidade $\text{H}^{-1}$) é dada por:
   $$\det(\mathbf{L}) = L_1 L_2 - M^2 = L_1 L_2 (1 - k^2)$$
   $$\mathbf{\Gamma} = \frac{1}{\det(\mathbf{L})} \begin{bmatrix} L_2 & -M \\ -M & L_1 \end{bmatrix}$$
   Para prevenir singularidade numérica quando $|k| = 1.0$, o solver aplica clamp $|k| \le 0.9999999$.

2. **Estampagem de Admitância AC (Regime Permanente Senoidal):**
   No domínio fasorial:
   $$\mathbf{I} = \frac{1}{j\omega} \mathbf{\Gamma} \mathbf{V} = \mathbf{Y}_{coupled} \mathbf{V}$$
   onde $\mathbf{Y}_{coupled} = -j \frac{\mathbf{\Gamma}}{\omega}$. A matriz de duas portas é estampada na matriz nodal MNA considerando os nós do primário $(p_1, n_1)$ e do secundário $(p_2, n_2)$:
   * Auto-admitância primário: $+Y_{11}$ em $(p_1, p_1)$ e $(n_1, n_1)$; $-Y_{11}$ em $(p_1, n_1)$ e $(n_1, p_1)$.
   * Auto-admitância secundário: $+Y_{22}$ em $(p_2, p_2)$ e $(n_2, n_2)$; $-Y_{22}$ em $(p_2, n_2)$ e $(n_2, p_2)$.
   * Acoplamento cruzado mútuo:
     $$\mathbf{A}[p_1, p_2] \mathrel{+}= Y_{12}, \quad \mathbf{A}[p_1, n_2] \mathrel{-}= Y_{12}, \quad \mathbf{A}[n_1, p_2] \mathrel{-}= Y_{12}, \quad \mathbf{A}[n_1, n_2] \mathrel{+}= Y_{12}$$
     $$\mathbf{A}[p_2, p_1] \mathrel{+}= Y_{21}, \quad \mathbf{A}[p_2, n_1] \mathrel{-}= Y_{21}, \quad \mathbf{A}[n_2, p_1] \mathrel{-}= Y_{21}, \quad \mathbf{A}[n_2, n_1] \mathrel{+}= Y_{21}$$

3. **Modelo Companheiro Transiente (Domínio do Tempo):**
   Integrando numericamente $\frac{d\mathbf{i}}{dt} = \mathbf{\Gamma} \mathbf{v}(t)$ com passo $\Delta t$:
   $$\mathbf{G}_{eq} = \begin{cases} \frac{\Delta t}{2} \mathbf{\Gamma} & \text{(Regra Trapezoidal)} \\ \Delta t \mathbf{\Gamma} & \text{(Backward Euler)} \end{cases}$$
   $$\mathbf{I}_{eq, k} = \begin{cases} \mathbf{i}_{k-1} + \mathbf{G}_{eq} \mathbf{v}_{k-1} & \text{(Regra Trapezoidal)} \\ \mathbf{i}_{k-1} & \text{(Backward Euler)} \end{cases}$$
   A matriz $\mathbf{G}_{eq}$ é estampada invariantemente no tempo na matriz $\mathbf{A}$, e o vetor de correntes históricas $\mathbf{I}_{eq}$ é injetado no vetor $\mathbf{z}$ a cada passo de tempo ($-I_{eq, 1}$ em $p_1$, $+I_{eq, 1}$ em $n_1$; $-I_{eq, 2}$ em $p_2$, $+I_{eq, 2}$ em $n_2$).

4. **Regime Permanente DC:**
   Em corrente contínua estática ($\omega = 0$), os enrolamentos de indutores ideais comportam-se como curto-circuitos perfeitos (condutância parasita $g = 10^6\ \Omega^{-1}$).

5. **Sintaxe SPICE Suportada:**
   * Acoplamento mútuo entre dois indutores: `K<name> <L1> <L2> <k>` (ex: `K1 L1 L2 0.999`)
   * Transformador integrado de 2 enrolamentos: `T<name> <p1> <n1> <p2> <n2> <L1> <L2> [k]` (ex: `T1 1 0 2 0 10m 2.5m 0.999`)

---

### 1.10 Redes de Duas Portas / Quadripolos (`two_port.py`)

Redes lineares invariantes no tempo com duas portas de acesso (Porta 1 de entrada e Porta 2 de saída) são caracterizadas por relações matriciais $2 \times 2$ entre as tensões $(V_1, V_2)$ e correntes $(I_1, I_2)$ de porta.

#### 1. Parâmetros de Impedância ($\mathbf{Z}$):
$$\begin{bmatrix} V_1 \\ V_2 \end{bmatrix} = \begin{bmatrix} z_{11} & z_{12} \\ z_{21} & z_{22} \end{bmatrix} \begin{bmatrix} I_1 \\ I_2 \end{bmatrix}$$
* $z_{11} = \left. \frac{V_1}{I_1} \right|_{I_2 = 0}$ (impedância de entrada em circuito aberto)
* $z_{12} = \left. \frac{V_1}{I_2} \right|_{I_1 = 0}$ (impedância de transferência reversa em circuito aberto)
* $z_{21} = \left. \frac{V_2}{I_1} \right|_{I_2 = 0}$ (impedância de transferência direta em circuito aberto)
* $z_{22} = \left. \frac{V_2}{I_2} \right|_{I_1 = 0}$ (impedância de saída em circuito aberto)

#### 2. Parâmetros de Admitância ($\mathbf{Y}$):
$$\begin{bmatrix} I_1 \\ I_2 \end{bmatrix} = \begin{bmatrix} y_{11} & y_{12} \\ y_{21} & y_{22} \end{bmatrix} \begin{bmatrix} V_1 \\ V_2 \end{bmatrix} = \mathbf{Z}^{-1} \begin{bmatrix} V_1 \\ V_2 \end{bmatrix}$$

#### 3. Parâmetros de Transmissão / ABCD ($\mathbf{T}$):
Convenção clássica de telecomunicações e linhas de transmissão (com corrente da porta 2 saindo do quadripolo):
$$\begin{bmatrix} V_1 \\ I_1 \end{bmatrix} = \begin{bmatrix} A & B \\ C & D \end{bmatrix} \begin{bmatrix} V_2 \\ -I_2 \end{bmatrix}$$
* $A = \frac{z_{11}}{z_{21}}$ (ganho de tensão inverso em circuito aberto, adimensional)
* $B = \frac{\det(\mathbf{Z})}{z_{21}}$ (impedância de transferência reversa em curto-circuito, em $\Omega$)
* $C = \frac{1}{z_{21}}$ (admitância de transferência reversa em circuito aberto, em $\text{S}$)
* $D = \frac{z_{22}}{z_{21}}$ (ganho de corrente inverso em curto-circuito, adimensional)

#### 4. Propriedades Estruturais e Operações:
* **Reciprocidade:** $z_{12} = z_{21} \iff AD - BC = 1 \iff y_{12} = y_{21}$.
* **Simetria:** $z_{11} = z_{22} \iff A = D \iff y_{11} = y_{22}$.
* **Impedância de Entrada sob Carga $Z_L$:** $Z_{in} = \frac{A Z_L + B}{C Z_L + D}$.
* **Impedância de Saída sob Gerador com Impedância $Z_S$:** $Z_{out} = \frac{D Z_S + B}{C Z_S + A}$.
* **Associação em Cascata:** Se dois quadripolos são ligados em cascata, a matriz de transmissão total é o produto matricial direto: $\mathbf{T}_{total} = \mathbf{T}_1 \mathbf{T}_2$.

---

### 1.11 Sistemas Trifásicos e Teorema de Blondel (`three_phase.py`)

A modelagem de circuitos polifásicos baseia-se na teoria de sistemas simétricos e balanceados:

#### 1. Tensões e Correntes Fasoriais:
Para uma fonte em estrela (Y) com sequência direta positiva (ABC) e referência de fase em $V_{an} = V_{LN} \angle 0^\circ$:
$$\mathbf{V}_{an} = V_{LN} \angle 0^\circ, \quad \mathbf{V}_{bn} = V_{LN} \angle -120^\circ, \quad \mathbf{V}_{cn} = V_{LN} \angle +120^\circ$$
As tensões de linha ($\mathbf{V}_{LL}$) relacionam-se por:
$$\mathbf{V}_{ab} = \mathbf{V}_{an} - \mathbf{V}_{bn} = \sqrt{3} V_{LN} \angle +30^\circ$$

#### 2. Cargas em Triângulo (Δ) e Estrela (Y):
* Equivalência $\Delta \leftrightarrow Y$: $Z_Y = \frac{Z_\Delta}{3}$.
* Corrente de linha em carga $\Delta$: $I_L = \sqrt{3} I_\Delta \angle -30^\circ$.

#### 3. Método dos Dois Wattímetros (Teorema de Blondel):
Para sistemas a 3 condutores sem neutro, a medição com dois wattímetros referenciados à linha comum C satisfaz:
$$W_1 = \text{Re}(\mathbf{V}_{ac} \cdot \mathbf{I}_a^*) = V_{LL} I_L \cos(30^\circ + \theta)$$
$$W_2 = \text{Re}(\mathbf{V}_{bc} \cdot \mathbf{I}_b^*) = V_{LL} I_L \cos(30^\circ - \theta)$$
Propriedades de síntese global de potência:
$$P_{3\phi} = W_1 + W_2 = \sqrt{3} V_{LL} I_L \cos\theta \quad [\text{W}]$$
$$Q_{3\phi} = \sqrt{3}(W_1 - W_2) = \sqrt{3} V_{LL} I_L \sin\theta \quad [\text{VAR}]$$
$$S_{3\phi} = \sqrt{P_{3\phi}^2 + Q_{3\phi}^2} = \sqrt{3} V_{LL} I_L \quad [\text{VA}]$$
$$\text{FP} = \frac{P_{3\phi}}{S_{3\phi}} = \cos\theta, \quad \tan\theta = \sqrt{3} \frac{W_1 - W_2}{W_1 + W_2}$$

---

## 2. 🛡️ Validação Topológica e Teoria dos Grafos (`validator.py`)

A maior causa de falha em simuladores de circuitos (inclusive no SPICE original) são circuitos fisicamente incoerentes que resultam em matrizes singulares (`LinAlgError`). O `pycircuit-solver` intercepta esses problemas **antes** de montar as matrizes através de algoritmos de Teoria dos Grafos:

```
[Netlist / Elementos] 
         ↓
 ┌─────────────────────────────────────────────────────────┐
 │                  CircuitValidator                       │
 │  1. Validação Léxica: valores > 0, nomes válidos        │
 │  2. Verificação de Terra: existência obrigatória de '0' │
 │  3. Conectividade BFS: detecção de sub-circuitos ilhados│
 │  4. Detecção DFS de Loops de Fontes de Tensão (KVL loop)│
 │  5. Verificação de Fontes de Controle de CCCS / CCVS    │
 └─────────────────────────────────────────────────────────┘
         ↓ (Circuito Topologicamente Seguro)
   [MNA Matrix Assembly]
```

### 2.1 Algoritmo BFS para Conectividade com o Terra
O circuito é interpretado como um grafo não-direcionado onde os nós do circuito são vértices e os elementos bifilares são arestas.
* Executa uma Busca em Largura (BFS) partindo do nó de referência `'0'`.
* Se algum nó não for visitado durante a travessia, esse nó pertence a uma "ilha flutuante" desconectada do terra. O validador rejeita a topologia com uma exceção explícita indicando exatamente o nó flutuante.

### 2.2 Algoritmo DFS para Detecção de Conflitos KVL (Loops de Tensão)
Duas fontes de tensão ideais em paralelo com valores diferentes, ou um anel fechado formado exclusivamente por fontes de tensão, violam a Lei de Kirchhoff das Tensões ($\sum V \neq 0$).
* O validador filtra o subgrafo composto exclusivamente por ramos de fontes ideais de tensão.
* Executa Busca em Profundidade (DFS) rastreando ciclos. Ao encontrar um ciclo de fontes de tensão, rejeita a simulação com erro topológico descritivo.

### 2.3 Tratamento de Sufixos de Engenharia
O módulo `parse_value` converte strings de notação científica e laboratorial em floats de SI:
* `1k` / `1K` $\to 1 \times 10^3$
* `4k7` $\to 4.7 \times 10^3$
* `10u` / `10uF` $\to 10 \times 10^{-6}$
* `1MEG` / `1meg` $\to 1 \times 10^6$
* `2.2m` / `2.2mA` $\to 2.2 \times 10^{-3}$

---

## 3. 🦀 Acelerador Híbrido em Rust (`rust_core`)

### 3.1 Por que Rust para o PyCircuit?
Em análises determinísticas de um único circuito (10 a 50 nós), o NumPy resolve o sistema linear em menos de $1\text{ ms}$, tornando o overhead imperceptível. No entanto, em **análises estatísticas de Monte Carlo**, o circuito precisa ser perturbado e resolvido **milhares de vezes** ($10.000$ a $100.000$ iterações).

No Python puro:
* O interpretador CPython possui o **GIL (*Global Interpreter Lock*)**, impedindo que threads Python executem computação numérica em paralelo real em processadores multi-core.
* A alocação contínua de objetos Python e dicionários de mapeamento de nós dentro do loop de Monte Carlo gera enorme pressão no Garbage Collector.

### 3.2 Arquitetura de Integração (PyO3 + Rayon + Nalgebra)
A camada Rust elimina esses gargalos operando como uma extensão nativa:

```
[Python User Space]
    Circuit, monte_carlo(circuit, n_samples=50_000)
         │
         ▼
[monte_carlo_rust.py] (Bridge Layer)
    - Converte rótulos de nós (strings) em inteiros 0..N
    - Empacota estruturas PyResistor, PyVSource, PyISource
         │  (Chamada CPython via PyO3 C-ABI)
         ▼
[rust_core (C-Extension compilada com LTO e Opt-Level 3)]
    ┌────────────────────────────────────────────────────────┐
    │  lib.rs (PyO3 interface)                               │
    ├────────────────────────────────────────────────────────┤
    │  rayon::par_iter                                       │
    │  ├── Thread 1: ChaCha8Rng (seed+1) → MNA LU Solve      │
    │  ├── Thread 2: ChaCha8Rng (seed+2) → MNA LU Solve      │
    │  ├── Thread 3: ChaCha8Rng (seed+3) → MNA LU Solve      │
    │  └── Thread N: ChaCha8Rng (seed+N) → MNA LU Solve      │
    ├────────────────────────────────────────────────────────┤
    │  nalgebra::full_piv_lu() (Pivotamento Total Estável)   │
    │  Estatísticas Nativas: Média, Desvio Padrão, P5, P95   │
    └────────────────────────────────────────────────────────┘
         │
         ▼ (Zero-Copy via numpy::IntoPyArray)
Retorno instantâneo para Python como NumPy ndarrays
```

### 3.3 Ganhos de Desempenho Medidos
Em testes automatizados no hardware de desenvolvimento (Apple Silicon M-series):
* **5.000 amostras (Ponte de Wheatstone):**
  * Python puro: $\approx 180\text{ ms}$
  * Rust multi-thread (`rayon`): $\approx 12\text{ ms}$
  * **Aceleração (*Speedup*): $\mathbf{15\times}$ mais rápido.**
* **50.000 amostras:**
  * Concluído em menos de $0.15\text{ s}$ em Rust.

### 3.4 Padrão de Resiliência: *Graceful Fallback*
O módulo `circuit_solver.engine.monte_carlo_rust` verifica a disponibilidade do binário compilado `pycircuit_core`. Caso o usuário esteja em um ambiente sem compilador Rust, o solver automaticamente utiliza o motor Python puro sem lançar exceções e sem alterar a interface pública da função `monte_carlo()`.

---

## 4. 🗂️ Estrutura e Responsabilidades dos Módulos

```
pycircuit-solver/
├── circuit_solver/
│   ├── core/
│   │   ├── elements.py         # Dataclasses de componentes (R, C, L, V, I, VCVS, VCCS, CCCS, CCVS, OpAmp, Diodos, BJT, MOS, Transformador)
│   │   ├── circuit.py          # Gestor de topologia e dicionários de mapeamento nodal
│   │   ├── validator.py        # Grafos (BFS/DFS), coerência de nós, loops ilegais e conversão de unidades
│   │   ├── solution.py         # Encapsulador do vetor solução com Tellegen e métodos V(node)
│   │   ├── subcircuit.py       # Expansão hierárquica e introspecção White-Box de subcircuitos
│   │   └── builtin_library.py  # Macromodelos didáticos embutidos (LM741, CMOS, Sallen-Key)
│   ├── engine/
│   │   ├── mna.py              # Montagem MNA unificada (estampas compartilhadas via _dc_stamp_entries para denso/esparso)
│   │   ├── ac.py               # Formulação fasorial em complex128 e resposta em frequência
│   │   ├── small_signal.py     # Análise CA de pequenos sinais com linearização de semicondutores
│   │   ├── sensitivity.py      # Análise de Sensibilidade Analítica (Rede Adjunta), WCA e EVA
│   │   ├── transient.py        # Motor transiente com cálculo UIC t=0 resistivo e auto-dispatch para Rust
│   │   ├── transient_rust.py   # Ponte Python ↔ Rust para simulação transiente ultra-rápida (>1M passos/s)
│   │   ├── nonlinear.py        # Semicondutores e solver não-linear Newton-Raphson com amortecimento
│   │   ├── dc_sweep.py         # Varredura paramétrica DC (.dc sweep) de fontes e resistências
│   │   ├── thevenin.py         # Algoritmo de injeção dual de Thévenin/Norton e P_max
│   │   ├── two_port.py         # Quadripolos / Redes de Duas Portas (Matrizes Z, Y, H, ABCD e cascata)
│   │   ├── three_phase.py      # Sistemas Trifásicos (Y-Δ) e Método dos Dois Wattímetros (Blondel)
│   │   ├── monte_carlo.py      # Implementação de referência Monte Carlo em Python puro
│   │   └── monte_carlo_rust.py # Ponte de alta performance para o crate Rust (Rayon) com suporte a transformador DC
│   ├── io/
│   │   └── parser.py           # Parser SPICE flexível (suporta arquivos e strings brutas, convenção I/G e PULSE/SIN)
│   ├── visualization/
│   │   └── plots.py            # Gráficos de Monte Carlo, Thévenin, Diagramas de Bode, Transiente e DC Sweep
│   └── cli.py                  # Interface CLI (solve, thevenin, ac, monte-carlo, tran, op, wca, explain, hierarchy, dc)
│
├── rust_core/                  # Crate Rust nativo (PyO3 + Rayon + Faer + Nalgebra)
│   ├── Cargo.toml              # Dependências de alta performance
│   └── src/
│       ├── lib.rs              # Definição do módulo PyO3 e exportações Python
│       ├── types.rs            # Structs pré-resolvidas sem overhead de strings
│       ├── mna_core.rs         # Montagem e decomposição LU densa e esparsa (Faer)
│       ├── monte_carlo.rs      # Rayon data-parallel worker para simulação Monte Carlo
│       └── transient.rs        # Motor de integração implícita transiente (>1M passos/s) com UIC resistivo
│
└── tests/                      # Suíte de 447 testes automatizados com pytest (437 aprovados, 10 skips estocásticos)
    ├── test_textbook_benchmarks.py     # 48 benchmarks analíticos canônicos (Sadiku, Nilsson, Boylestad, Dorf, Sedra, Hayt, Irwin)
    ├── test_two_port.py                # Redes de duas portas (Z, Y, H, ABCD, Zin, Zout e cascata)
    ├── test_three_phase.py             # Sistemas trifásicos Y-Y, Y-Δ e Teorema de Blondel (dois wattímetros)
    ├── test_stress.py                  # Bateria de stress, carga extrema (400 nós, R-2R 64-bit, 25k Monte Carlo) e fuzzing
    ├── test_random_dc_cross.py         # Teste cruzado aleatório estocástico (denso x esparso x Rust em 60 circuitos)
    ├── test_rust_sparse_transformer.py # Validação de transformadores em DC no backend esparso Rust
    ├── test_transient_convergence.py   # Ordem de convergência Trapezoidal 2ª ordem vs Euler 1ª ordem e UIC em t=0
    ├── test_spice_convention.py        # Convenção SPICE para I e G e derivação do ponto de operação DC de PULSE/SIN
    ├── test_netlist_input.py           # Suporte a netlist como texto bruto em memória e caminhos de arquivo
    ├── test_sensitivity_wca.py        # Sensibilidade analítica (Rede Adjunta), EVA, Taylor e RSS
    ├── test_rust_transient.py          # Transiente em Rust, conservação de energia e speedup
    ├── test_small_signal_ac.py         # Análise de pequenos sinais com Diodos, BJT e MOSFET
    ├── test_subcircuits.py             # Subcircuitos (.subckt), parâmetros, fórmulas e introspecção
    ├── test_coupled_inductors.py       # Transformadores e acoplamento magnético mútuo (elemento K)
    ├── test_nonlinear.py               # Newton-Raphson e semicondutores (Diodo, Zener, BJT, MOS)
    ├── test_dc_sweep.py                # Varredura paramétrica DC e famílias de curvas características
    ├── test_transient.py               # Transiente analítico RC, RL, RLC e formas de onda PULSE/SIN
    ├── test_rust_monte_carlo.py        # Correção, sementes e paralelismo do motor Rust
    ├── test_sparse_and_rust.py         # MNA esparso (Faer / SciPy) e circuitos gigantes (>1000 nós)
    ├── test_robustness_edge_cases.py   # Grafos BFS, laços ilegais KVL, R_th=0 e parser
    ├── test_validator.py               # Validação topológica, terra '0' e sufixos de engenharia
    ├── test_mna_engine.py              # Motor MNA estático e 4 fontes dependentes
    ├── test_ac_engine.py               # Análise CA fasorial, filtros e ressonância
    └── test_cli.py                     # Validação completa de todos os subcomandos da CLI
```

---

## 5. 📊 Matriz de Validação Multi-Livros e Suíte de Testes (447 Testes)

A biblioteca é auditada de ponta a ponta por uma suíte de **521 testes automatizados** (511 aprovados, 10 casos numericamente mal condicionados ignorados com diagnóstico) executados via `pytest`, cobrindo **48 exercícios canônicos clássicos** com referências analíticas e aproximações explicitadas, inspirados em 7 obras bibliográficas da Engenharia Elétrica e Microeletrônica:

1. **Alexander & Sadiku (*Fundamentos de Circuitos Elétricos*):**
   * Divisores e reduções série-paralelo (Cap. 2, Ex. 2.8).
   * Transformação estrela-triângulo ($\Delta - Y$) em ponte (Cap. 2, Ex. 2.15).
   * Análise nodal com supernós e fontes de corrente (Cap. 3, Ex. 3.1 & 3.3).
   * Análise com fonte controlada VCCS (Cap. 3, Ex. 3.2).
   * Supermalha (Supermesh) com fonte de corrente compartilhada entre malhas (Cap. 3, Ex. 3.7).
   * Teorema de Thévenin, Norton e Máxima Transferência de Potência (Cap. 4, Ex. 4.8 & 4.9).
   * Circuitos com Op-Amp: Inversor, Não-Inversor, Somador e Subtrator (Cap. 5, Ex. 5.2, 5.5, 5.6 & 5.7).
   * Amplificador integrador ideal no domínio do tempo (Cap. 6, Ex. 6.13).
   * Amplificador diferenciador ideal em regime permanente CA fasorial (Cap. 6).
   * Resposta transitória ao degrau de circuito RLC série superamortecido ($\alpha > \omega_0$) (Cap. 8, Ex. 8.4).
   * Resposta natural transitória de circuito RLC paralelo criticamente amortecido ($\alpha = \omega_0$) (Cap. 8, Ex. 8.6).
   * Regime permanente CA fasorial RLC série (Cap. 9, Ex. 9.9).
   * Correção de fator de potência com banco de capacitores em paralelo (Cap. 11, Ex. 11.15).
   * Medição de potência trifásica pelo Método dos Dois Wattímetros (Teorema de Blondel) (Cap. 12, Ex. 12.14).
   * Circuitos magneticamente acoplados e transformador linear (Cap. 13, Ex. 13.1).
2. **Nilsson & Riedel (*Circuitos Elétricos*):**
   * Equivalente de Thévenin com fonte dependente mista (Cap. 4, Ex. 4.10).
   * Somador inversor com ponderação de ganhos arbitrários (Cap. 5, Ex. 5.3).
   * Resposta transitória ao degrau DC de circuito RL de 1ª ordem ($i_L(t) = I_\infty(1 - e^{-t/\tau})$) (Cap. 7, Ex. 7.4).
   * Circuito RLC paralelo em regime CA de alta frequência (Cap. 9, Ex. 9.6).
   * Filtro ativo de 2ª ordem passa-baixas Sallen-Key com aproximação Butterworth pura (Cap. 14).
3. **Boylestad (*Introdução à Análise de Circuitos & Dispositivos Eletrônicos*):**
   * Regulador de tensão com Diodo Zener e sensibilidade à variação de carga (Cap. 2).
   * Amplificador BJT emissor comum com degeneração de emissor $R_E$ e polarização por divisor (Cap. 5).
   * Ponte de Wheatstone em condição de equilíbrio de nulo (Cap. 6).
   * Rede escada R-2R de 4 bits para conversão Digital/Analógica (Cap. 8).
   * Teorema de Millman para baterias em paralelo com resistência interna (Cap. 9).
4. **Dorf & Svoboda (*Introdução aos Circuitos Elétricos*):**
   * Op-Amp com rede de realimentação em T para altíssimo ganho estável (Cap. 6).
   * Buffer seguidor de tensão e isolamento de impedância de carga (Cap. 6).
   * Resposta natural transitória de circuito RLC paralelo subamortecido ($\omega_d = \sqrt{\omega_0^2 - \alpha^2}$) (Cap. 8).
5. **Sedra & Smith (*Microelectronic Circuits / Microeletrônica*):**
   * Amplificador de transimpedância com conversão corrente-tensão $I \to V$ com Op-Amp (Cap. 2).
   * Ponto Q e reta de carga DC de diodo retificador PN com modelo de Shockley (Cap. 4).
   * Retificador de onda completa em ponte (Ponte de Graetz) simétrica (Cap. 4).
   * Amplificador Fonte Comum MOSFET NMOS operando na saturação quadrática (Cap. 5).
   * Polarização de transistor BJT NPN por divisor de 4 resistores na região ativa (Cap. 6).
   * Espelho de corrente BJT (Current Mirror) com compensação de corrente de base (Cap. 7).
   * Inversor lógico digital CMOS estático com par complementar NMOS/PMOS (Cap. 14).
6. **Hayt, Kemmerly & Durbin (*Engineering Circuit Analysis / Análise de Circuitos em Engenharia*):**
   * Teorema e Princípio da Superposição com 3 fontes independentes simultâneas (Cap. 5).
   * Resposta transitória analítica de carga RC de 1ª ordem ($v_C(t) = V_s(1 - e^{-t/\tau})$) (Cap. 7).
   * Circuito RLC série em ressonância e fator de qualidade $Q$ com ganho de tensão reativa (Cap. 9).
   * Acoplamento magnético e reflexão de impedância de carga ($Z_{ref} = R_L / n^2$) no circuito primário (Cap. 13).
   * Extração completa dos parâmetros de Impedância ($Z$) e Transmissão ($ABCD$) de Quadripolos em T (Cap. 17).
7. **Irwin & Nelms (*Basic Engineering Circuit Analysis / Análise Básica de Circuitos*):**
   * Análise Nodal Modificada com fonte de corrente controlada por corrente (CCCS - Elemento F) (Cap. 3).
   * Amplificador de Instrumentação (INA) de 3 Op-Amps com ganho diferencial calibrado (Cap. 4).
   * Ponte de Wheatstone desbalanceada e extração exata do equivalente de Thévenin (Cap. 5).
8. **Casos Degenerados, Não-Lineares e Robustez:**
   * Redes com $R_{th} = 0\ \Omega$ (fonte ideal nos terminais) e $R_{th} \to \infty$ (circuito aberto).
   * Detecção topológica de loops fechados de fontes ideais de tensão via DFS.
   * Isolamento e poda de subcircuitos desconectados sem caminho para o terra via BFS.
   * Tolerância do parser SPICE para quebras de linha com o operador `+`.
   * Convergência quadrática estrita de Newton-Raphson com semicondutores.
   * Conservação de energia e balanço de Joule em regime transitório.
   * Análise de sensibilidade exata por rede adjunta versus diferenças finitas.

---

## 6. 🚀 Como Utilizar o Sistema

### 6.1 Via Código Python
```python
from circuit_solver import Circuit, VoltageSource, Resistor, solve
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.engine.monte_carlo_rust import monte_carlo

# 1. Definir circuito
circuit = Circuit([
    VoltageSource("V1", "1", "0", 24.0),
    Resistor("R1", "1", "2", 10.0, tolerance=0.05),
    Resistor("R2", "2", "0", 20.0, tolerance=0.05),
    Resistor("Rload", "2", "0", 30.0)
])

# 2. Resolução Nodal
sol = solve(circuit)
print(f"Tensão no nó 2: {sol.V('2'):.4f} V")
print(f"Auditoria de Potência: {sol.power_audit()['balance']:.2e} W")

# 3. Equivalente de Thévenin entre nó 2 e terra
th = thevenin_norton(circuit, node_a="2", node_b="0")
print(f"Vth = {th.v_th:.2f} V, Rth = {th.r_th:.2f} Ohms, Pmax = {th.p_max:.2f} W")

# 4. Simulação Estatística Acelerada em Rust
mc = monte_carlo(circuit, n_samples=10000, seed=42)
print(mc.summary())
```

### 6.2 Via Linha de Comando (CLI)
```bash
# Resolver netlist SPICE linear ou não-linear (DC)
pycircuit solve circuito.cir

# Ponto de operação DC (.op) e linearização para pequenos sinais
pycircuit op amplificador.cir

# Análise de Sensibilidade e Pior Caso (Worst-Case / EVA)
pycircuit sensitivity circuito.cir --target out
pycircuit wca circuito.cir -t out --tol 0.05

# Extrair equivalente de Thévenin e Norton com curva de máxima potência
pycircuit thevenin circuito.cir -a 2 -b 0 --plot

# Análise de Monte Carlo acelerada em Rust com histograma
pycircuit monte-carlo circuito.cir -n 10000 -t 0.05 --plot

# Resposta em frequência CA (Diagrama de Bode)
pycircuit ac filtro.cir --sweep 10 100000 100 --node out --plot

# Análise transitória no domínio do tempo (backend Rust de alta velocidade)
pycircuit tran circuito.cir --tstop 10m --dt 10u --backend rust --plot

# Relatório didático White-Box e matrizes MNA de subcircuitos
pycircuit explain circuito.cir

# Árvore ASCII de hierarquia de subcircuitos
pycircuit hierarchy circuito.cir

# Varredura paramétrica DC (.dc sweep)
pycircuit dc circuito.cir --sweep V1 0 10 0.1 --node out --plot
```

---

## 7. 🦀 Núcleo Rust Expandido e Solvers Esparsos para Circuitos Gigantes

### 7.1 Expansão do Núcleo Rust para 100% dos Elementos Lineares
O crate `pycircuit_core` (em Rust) foi expandido para cobrir **100% dos elementos lineares** suportados pelo framework:
* **Resistores** com tolerâncias estatísticas gaussianas.
* **Fontes Independentes:** Tensão (`V`) e Corrente (`I`).
* **Fontes Dependentes / Controladas:**
  * VCVS ($E$): Fonte de tensão controlada por tensão ($\mu$).
  * VCCS ($G$): Fonte de corrente controlada por tensão ($g_m$).
  * CCCS ($F$): Fonte de corrente controlada por corrente ($\beta$).
  * CCVS ($H$): Fonte de tensão controlada por corrente ($r_m$).
* **Amplificadores Operacionais Ideais:** Elemento $U$/$A$ com curto-circuito virtual ($V_{in+} - V_{in-} = 0$) e ramo de saída auxiliar.
* **Indutores em DC:** Estampados como curto-circuito ideal com condutância parasita controlada ($10^6\ \Omega^{-1}$).

Com isso, circuitos ativos complexos (filtros ativos, amplificadores de instrumentação com Op-Amps, circuitos a transistores em pequenos sinais com fontes controladas) rodam diretamente no kernel multi-thread Rayon em Rust, executando 10.000 amostras de Monte Carlo em milissegundos sem recorrer ao fallback Python.

### 7.2 Solvers Esparsos para Circuitos Gigantes (> 1.000 nós)
Para redes de grande porte (malhas de distribuição, chips VLSI, escadas R-2R de alta resolução com centenas ou milhares de nós):
1. **Montagem Esparsa Triplet/COO:**
   * `build_mna_sparse(circuit)` coleta as estampas em formato de coordenadas (COO) e as converte diretamente para matriz de colunas comprimidas (`scipy.sparse.csc_matrix`), somando condutâncias paralelas automaticamente em complexidade $O(N)$.
2. **Resolução com SuperLU no Python:**
   * Utiliza `scipy.sparse.linalg.spsolve`, reduzindo o consumo de memória de $O(N^2)$ para $O(N)$ e o tempo de execução de segundos para frações de milissegundo.
3. **Solver Esparso Nativo em Rust (`faer`):**
   * Implementado em `mna_core::solve_mna_sparse` e exposto via `pycircuit_core.solve_mna_sparse`.
   * Realiza fatoração simbólica e numérica $LU$ esparsa com suporte a instruções SIMD nativas e paralelismo de thread pool, resolvendo circuitos com mais de 1.000 nós em menos de 50 ms.
4. **Despacho Automático (`solver="auto"`):**
   * O solver `solve(circuit, solver="auto", sparse_threshold=500)` automaticamente ativa o solver esparso quando o número de incógnitas atinge o limiar configurado (padrão 500 nós).

---

## 8. 🧩 Suporte a Subcircuitos SPICE Hierárquicos e Mecanismo White-Box Pedagógico

### 8.1 Compatibilidade Industrial Padrão SPICE (.subckt / .ends e Elementos X)
O `pycircuit` implementa o mecanismo de modularização hierárquica compatível com as ferramentas comerciais de EDA (SPICE, LTspice, HSPICE, PSpice):
1. **Definição de Macromodelos:**
   Blocos reutilizáveis definidos por `.SUBCKT <nome> <pinos...> [PARAMS: p1=v1 ...]`, encerrados por `.ENDS [<nome>]`.
2. **Instanciação Modular:**
   Instâncias `X<nome> <nós_conectados...> <modelo> [PARAMS: p1=v1 ...]`.
3. **Avaliação Paramétrica e Expressões Matemáticas:**
   Suporte a parâmetros padrão, sobrescrita de instância e fórmulas matemáticas SPICE entre chaves `{...}` (ex: `{1 / (2 * pi * FC * C)}`, `{R1 * 2}`), com suporte a sufixos de engenharia (`k`, `u`, `n`, `p`, `MEG`, etc.) e funções elementares (`sin`, `cos`, `sqrt`, `exp`, etc.).
4. **Subcircuitos Aninhados Recursivos (*Nested Subcircuits*):**
   Suporte a múltiplos níveis de hierarquia (subcircuitos que instanciam outros subcircuitos), com qualificação correta de escopo (`X_BLOCO.X_ESTAGIO.R1`).
5. **Diretiva de Inclusão de Arquivos Externos:**
   Suporte a `.INCLUDE "<arquivo.lib>"` e `.INC "<arquivo.cir>"`, permitindo carregar modelos de fabricantes comerciais e bibliotecas acadêmicas.

### 8.2 Arquitetura Pedagógica de Introspecção (*White-Box Hierarchy Engine*)
Projetado especificamente para pesquisa acadêmica, dissertações de mestrado e ensino de Engenharia Elétrica, eliminando o comportamento "caixa-preta" (*black box*) dos simuladores proprietários:
1. **Árvore Hierárquica Inspecionável (`HierarchyTree`):**
   Acesso em tempo de execução via Python (`circuit.hierarchy_tree()`) e via linha de comando (`pycircuit hierarchy circuito.cir`).
2. **Relatório Didático Passo a Passo (`circuit.explain_hierarchy()` / `pycircuit explain`):**
   * Tabela visual "Pino Formal $\rightarrow$ Nó Global".
   * Resolução detalhada de cada parâmetro e fórmula.
   * Lista de nós internos privados criados (`X1.int1`).
   * **Análise de Impacto Dimensional MNA:** Quantifica exatamente quantos nós nodais e variáveis de corrente auxiliar cada subcircuito adicionou e o crescimento da ordem da matriz $\mathbf{A}$.
3. **Sondagem de Grandezas Internas (*Scoped Probing*):**
   Permite medir grandezas internas de qualquer bloco com notação de ponto: `sol.V("X1.diff_pair")` ou gerar raio-X completo do bloco com `sol.subcircuit_summary("X1")`.

### 8.3 Biblioteca Pedagógica Embutida (*Textbook Library*)
O módulo `circuit_solver.core.builtin_library` disponibiliza macromodelos de livros-texto consagrados (Alexander & Sadiku, Sedra & Smith):
* `OPAMP_741`: Macromodel clássico do LM741 com resistência de entrada diferencial ($R_{in} = 2\text{ M}\Omega$), ganho em malha aberta finito ($A_{OL} = 200.000$) e impedância de saída ($R_{out} = 75\ \Omega$).
* `OPAMP_IDEAL`: Curto-circuito virtual ideal com ganho infinito.
* `RC_LOWPASS`: Filtro passa-baixa paramétrico de 1ª ordem.
* `SALLEN_KEY_LP`: Filtro ativo passa-baixas de 2ª ordem com topologia Sallen-Key.
* `CMOS_INVERTER`: Célula lógica digital inversora CMOS com par complementar NMOS/PMOS.
* `BJT_DIFF_PAIR`: Par diferencial básico a transistor bipolar com resistores de cauda e carga.

---

## 9. 🔬 Análise CA de Pequenos Sinais com Semicondutores e Ponto de Operação (.op)

A análise fasorial em regime permanente senoidal clássica aplica-se a redes estritamente lineares ($R, L, C$). Para amplificadores de áudio, rádio-frequência (RF) e circuitos com transistores, o `pycircuit` implementa a metodologia rigorosa em duas etapas utilizada nos simuladores industriais e nos livros de Microeletrônica (Sedra & Smith, Neamen, Boylestad):

### 9.1 Algoritmo em Duas Etapas (Ponto Q DC + Linearização Incremental)

1. **Passo 1: Ponto de Operação DC Estático ($Q$-Point — Newton-Raphson):**
   * Fontes de sinal CA são desativadas temporariamente ($V_{ac} = 0, I_{ac} = 0$).
   * Capacitores comportam-se como circuitos abertos ($g = 0$) e indutores como curto-circuitos ($g = 10^6\ \Omega^{-1}$).
   * O solver de Newton-Raphson com amortecimento `pnjlim` e `fetlim` converge para as tensões dos nós $\mathbf{V}_{DC}$ e as correntes estáticas de cada semicondutor ($I_D, I_C, I_B, I_D$).

2. **Passo 2: Modelagem Incremental de Pequenos Sinais (Jacobiano Analítico):**
   No ponto $Q$, cada semicondutor é substituído pelo seu circuito equivalente linear:
   * **Diodo / Zener:**
     $$r_d = \frac{n V_T}{I_{D0} + I_S}, \quad g_d = \frac{1}{r_d}$$
   * **Transistor Bipolar (BJT — Modelo Híbrido-$\pi$ de Giacoleto):**
     * Transcondutância: $g_m = \frac{|I_{C0}|}{V_T}$
     * Resistência de entrada base-emissor: $r_\pi = \frac{\beta}{g_m} = \frac{V_T}{|I_{B0}|}$
     * Resistência de saída de coletor: $r_o = \frac{V_A + |V_{CE0}|}{|I_{C0}|}$
     * Estampa CA: Resistor $r_\pi$ entre base e emissor, resistor $r_o$ entre coletor e emissor, e fonte de corrente controlada por tensão (VCCS) $g_m \cdot v_{be}$ fluindo do coletor para o emissor.
   * **Transistor de Efeito de Campo (MOSFET):**
     * Na saturação ($V_{DS} \ge V_{GS} - V_{th}$):
       $$g_m = K_p (V_{GS} - V_{th}) (1 + \lambda V_{DS}) = \sqrt{2 K_p I_D (1 + \lambda V_{DS})}, \quad r_{ds} = \frac{1}{\lambda I_D}$$
     * No triodo ($V_{DS} < V_{GS} - V_{th}$):
       $$g_m = K_p V_{DS} (1 + \lambda V_{DS}), \quad r_{ds} = \frac{1}{g_{ds}}$$
     * Estampa CA: Fonte VCCS $g_m \cdot v_{gs}$ do dreno para a fonte e resistor de saída $r_{ds}$.

3. **Passo 3: Superposição de Fontes e Resolução Fasorial AC:**
   * Fontes CC puras de polarização ($V_{CC}, V_{DD}$) tornam-se terra CA ($V_{ac} = 0$).
   * Fontes CA independentes fornecem a excitação senoidal complexa $V_{ac} e^{j\theta}$.
   * O sistema complexo $\mathbf{A}_{AC}(j\omega) \mathbf{X}(j\omega) = \mathbf{Z}_{AC}$ é montado e resolvido diretamente para cada frequência da varredura (`ac_sweep`), traçando diagramas de Bode de amplificadores Emissor Comum, Coletor Comum e Fonte Comum.

### 9.2 Relatório Pedagógico do Ponto de Operação (`pycircuit op`)
Permite ao estudante ou pesquisador inspecionar instantaneamente:
* Tensão nos terminais e polarização de junção ($V_D, V_{BE}, V_{CE}, V_{GS}, V_{DS}$).
* Identificação automática da região de operação: **Corte**, **Saturação** ou **Ativa Direta / Triodo**.
* Parâmetros calculados do modelo equivalente ($g_m, r_\pi, r_o, r_d, r_{ds}$).

---

## 10. ⚡ Motor Transiente em Rust com Auditoria de Energia e Alta Performance

Para simulações no domínio do tempo com dezenas de milhares de passos temporais (ex: osciladores, fontes comutadas, conversores de potência, trens de pulso de alta frequência), o `pycircuit` introduz o **Solver Transiente Nativo em Rust** (`pycircuit_core.solve_transient_rust`), combinando performance em nível C/Rust com rigor acadêmico de conservação de energia.

### 10.1 Formulação Matemática dos Modelos Companion

Para elementos dinâmicos lineares ($C$, $L$, Transformadores/Indutores Mútuos), as equações diferenciais constitutivas $i_C = C \frac{dv_C}{dt}$ e $v_L = L \frac{di_L}{dt}$ são discretizadas por esquemas de integração implícitos estáveis:

1. **Regra Trapezoidal de 2ª Ordem ($A$-estável, sem dissipação numérica artificial):**
   * **Capacitor:**
     $$G_{eq} = \frac{2 C}{\Delta t}, \quad I_{eq}[k] = G_{eq} \cdot v_C[k-1] + i_C[k-1]$$
   * **Indutor:**
     $$G_{eq} = \frac{\Delta t}{2 L}, \quad I_{eq}[k] = i_L[k-1] + G_{eq} \cdot v_L[k-1]$$
2. **Backward Euler de 1ª Ordem ($L$-estável, altamente amortecido):**
   * **Capacitor:** $G_{eq} = \frac{C}{\Delta t}, \quad I_{eq}[k] = G_{eq} \cdot v_C[k-1]$
   * **Indutor:** $G_{eq} = \frac{\Delta t}{L}, \quad I_{eq}[k] = i_L[k-1]$

### 10.2 Otimização Algorítmica e Complexidade Computacional

Em circuitos lineares com passo temporal fixo $\Delta t$:
* A matriz estendida MNA $\mathbf{A}$ é **invariante no tempo**.
* **Fatoração Prévia $O(N^3)$:** Realizada uma única vez antes de iniciar o loop temporal via Decomposição LU em Rust (`nalgebra::DMatrix::lu`).
* **Loop de Passo Temporal $O(N^2)$:** Para cada instante $t_k$, o solver apenas avalia as fontes $Z(t_k)$, calcula as correntes companion $I_{eq}$ e resolve por forward/backward substitution:
  $$\mathbf{x}[k] = \mathbf{U}^{-1} (\mathbf{L}^{-1} (\mathbf{P} \cdot \mathbf{z}[k]))$$
* **Desempenho Real:** Atinge taxas superiores a **1.000.000 de passos por segundo**, reduzindo simulações de 50.000 passos de segundos para poucos milissegundos.

### 10.3 Auditoria Pedagógica de Energia e Conservação Física

Como parte do compromisso de transparência *White-Box* e fundamentação para pesquisas acadêmicas, o motor calcula simultaneamente em Rust a cada passo:

1. **Energia Instantânea Armazenada nos Campos Reativos ($E(t)$):**
   $$E_{\text{stored}}(t) = \frac{1}{2} \sum_{i} C_i \cdot v_{C_i}^2(t) + \frac{1}{2} \sum_{j} L_j \cdot i_{L_j}^2(t)$$
2. **Taxa de Dissipação Joule Instantânea em Resistores ($P_{\text{diss}}(t)$):**
   $$P_{\text{diss}}(t) = \sum_{k} \frac{v_{R_k}^2(t)}{R_k}$$
3. **Balanço Energético:** Permite verificar graficamente que a energia líquida entregue pelas fontes é exatamente convertida em calor Joule e trabalho de armazenamento de campo magnético e eletrostático ($\int P_{\text{fonte}} dt = \Delta E_{\text{stored}} + \int P_{\text{diss}} dt$).

### 10.4 Despacho Automático e Uso via CLI / Python

O parâmetro `backend` em `solve_transient` controla o despacho:
* `"auto"` (Padrão): Utiliza o motor compilado em Rust se disponível e a topologia for linear; caso haja funções Python dinâmicas arbitrárias ou semicondutores não linearizados, realiza fallback transparente para o solver Python/SciPy.
* `"rust"`: Execução forçada em Rust (garante máxima performance e emite erro explicativo se houver elementos incompatíveis).
* `"python"`: Execução pura em SciPy para propósitos de comparação analítica.

Na CLI:
```bash
pycircuit tran netlist.cir --tstop 10m --dt 10u --backend rust --plot
```

---

## 11. 🎯 Análise de Sensibilidade Analítica via Rede Adjunta e Pior Caso (Worst-Case / EVA)

Em projetos de engenharia eletrônica de alta confiabilidade (aeroespacial, médica, automotiva ISO 26262 e certificações militares), a análise nominal é insuficiente. O projetista e o pesquisador acadêmico necessitam quantificar:
1. **Sensibilidade Paramétrica:** O quanto uma variação em cada componente altera a tensão do nó crítico.
2. **Pior Caso Extremo (Worst-Case / EVA - Extreme Value Analysis):** Resolução dos dois cantos indicados pelas derivadas nominais; não prova extremos globais.
3. **Pior Caso Estatístico (RSS - Root-Sum-Square):** Combinação de primeira ordem das tolerâncias; não implica automaticamente 3σ.

### 11.1 Formulação Matemática do Método da Rede Adjunta (*Adjoint Network Method*)

O `pycircuit` implementa o clássico **Método da Rede Adjunta** (Tellegen / Director & Rohrer, 1969), que calcula a sensibilidade analítica exata de um nó de interesse $k$ com respeito a **todos os $M$ componentes do circuito simultaneamente com custo de um único solve adicional**:

Dado o sistema linear MNA do circuito nominal:
$$\mathbf{A} \mathbf{x} = \mathbf{z}$$

Diferenciando implicitamente com respeito a qualquer parâmetro $p$:
$$\mathbf{A} \frac{\partial \mathbf{x}}{\partial p} + \frac{\partial \mathbf{A}}{\partial p} \mathbf{x} = \frac{\partial \mathbf{z}}{\partial p} \implies \frac{\partial \mathbf{x}}{\partial p} = \mathbf{A}^{-1} \left( \frac{\partial \mathbf{z}}{\partial p} - \frac{\partial \mathbf{A}}{\partial p} \mathbf{x} \right)$$

Para a tensão no nó alvo $k$, $V_k = \mathbf{e}_k^T \mathbf{x}$. Definindo o vetor adjunto $\mathbf{y}_k$ como a solução do sistema linear transposto:
$$\mathbf{A}^T \mathbf{y}_k = \mathbf{e}_k \implies \mathbf{y}_k^T = \mathbf{e}_k^T \mathbf{A}^{-1}$$

A sensibilidade analítica em relação a $p$ é obtida diretamente pelo produto interno:
$$\frac{\partial V_k}{\partial p} = \mathbf{y}_k^T \left( \frac{\partial \mathbf{z}}{\partial p} - \frac{\partial \mathbf{A}}{\partial p} \mathbf{x} \right)$$

#### Estampas Analíticas por Componente:
* **Resistor $R$ (entre os nós $p$ e $n$):**
  $$\frac{\partial V_k}{\partial R} = \frac{(y_p - y_n)(x_p - x_n)}{R^2}$$
  onde $(x_p - x_n)$ é a tensão nominal no resistor e $(y_p - y_n)$ é a tensão no elemento no circuito adjunto.
* **Fonte de Tensão Independente $V_s$:**
  $$\frac{\partial V_k}{\partial V_s} = y_{aux}$$
* **Fonte de Corrente Independente $I_s$ (API: injeta em $p$, retira de $n$):**
  $$\frac{\partial V_k}{\partial I_s} = y_p - y_n$$
* **Fonte Controlada VCVS ($E$, ganho $\mu$):**
  $$\frac{\partial V_k}{\partial \mu} = y_{aux} \cdot (x_{cp} - x_{cn})$$

### 11.2 Sensibilidade Relativa Normalizada (Bode Sensitivity)

Para comparar grandezas com unidades diferentes ($\Omega$, V, A), calcula-se a sensibilidade normalizada percentual:
$$S_p^{V_k} = \frac{p}{V_k} \frac{\partial V_k}{\partial p} = \frac{\partial \ln V_k}{\partial \ln p}$$
*Exemplo:* $S_R^{V_k} = +0.75$ indica que uma variação de $+1\%$ na resistência provoca uma elevação de $+0.75\%$ na tensão de saída.

### 11.3 Métodos de Pior Caso: EVA vs Taylor vs RSS

Com a derivada $\frac{\partial V_k}{\partial p}$ e a tolerância de fabricação $\tau_i$ de cada componente:

1. **Determinação dos Cantos de Pior Caso (Extreme Value Analysis - EVA):**
   * Se $\frac{\partial V_k}{\partial p_i} > 0 \implies$ Para maximizar $V_k$, toma-se $p_i^{\max} = p_i + |p_i|\tau_i$; para minimizar, $p_i^{\min} = p_i - |p_i|\tau_i$.
   * Se $\frac{\partial V_k}{\partial p_i} < 0 \implies$ Para maximizar $V_k$, toma-se $p_i^{\min} = p_i - |p_i|\tau_i$; para minimizar, $p_i^{\max} = p_i + |p_i|\tau_i$.
   * **Re-solve Não-Linear Exato:** O `pycircuit` monta os circuitos nos cantos calculados e os resolve via MNA, capturando não-linearidades de curvatura reais da rede elétrica.

2. **Aproximação Linear de Taylor de 1ª Ordem:**
   $$\Delta V_{\text{linear}} = \sum_i \left| \frac{\partial V_k}{\partial p_i} \right| \cdot (\tau_i |p_i|)$$
   $$V_k^{\text{lin}} \in [V_{\text{nom}} - \Delta V_{\text{linear}}, \ V_{\text{nom}} + \Delta V_{\text{linear}}]$$

3. **Combinação Linear RSS (sem cobertura probabilística implícita):**
   $$\Delta V_{\text{RSS}} = \sqrt{\sum_i \left( \frac{\partial V_k}{\partial p_i} \cdot \tau_i p_i \right)^2}$$
   $$V_k^{\text{RSS}} \in [V_{\text{nom}} - \Delta V_{\text{RSS}}, \ V_{\text{nom}} + \Delta V_{\text{RSS}}]$$

### 11.4 Relatório Pedagógico e Pareto de Sensibilidade

O método `res.summary()` e o ranking `res.pareto_ranking()` decompõem o impacto absoluto linear total na saída, identificando quais componentes são os "vilões da tolerância" que mais afetam a especificação do circuito.

#### Exemplo via Python:
```python
from circuit_solver import Circuit, VoltageSource, Resistor, analyze_sensitivity

circ = Circuit([
    VoltageSource("V1", "1", "0", 10.0),
    Resistor("R1", "1", "2", 1000.0, tolerance=0.05),
    Resistor("R2", "2", "0", 1000.0, tolerance=0.05),
])

res = analyze_sensitivity(circ, target_node="2")
print(res.summary())
```

#### Exemplo via Terminal (CLI):
```bash
pycircuit sensitivity netlist.cir --target 2
# ou usando o alias rápido
pycircuit wca netlist.cir -t out --tol 0.01
```





### Interpretação para estudo

RSS = √Σ(∂V/∂p · Δp)² só corresponde a 3σ quando cada Δp é 3σ e as variações são independentes no modelo linear. Para distribuições uniformes independentes em ±Δp, σ(V) ≈ RSS/√3; correlações exigem covariâncias. O Pareto é a fração do impacto absoluto linear. Falhas ao resolver um canto são propagadas, sem substituição silenciosa por Taylor.

No transitório Python, semicondutores não são simulados: são rejeitados explicitamente. Indutores acoplados usam a matriz de indutâncias e energia E=½L1·i1²+M·i1·i2+½L2·i2². No método dos dois wattímetros, para fasores RMS e três fios, S=Vac·conj(Ia)+Vbc·conj(Ib), inclusive com cargas desequilibradas e sequência inversa. Com neutro conduzindo, use as três potências fase–neutro.


### Inicialização transitória e fontes controladas por corrente

Python e Rust propagam falhas do ponto DC inicial; iniciar com estado zero exige a escolha explícita de `uic=True` e condições iniciais compatíveis. O ponto inicial sem UIC usa os valores das formas de onda em t=0. O modelo DC atual aproxima cada enrolamento por 1 µΩ; não é um curto ideal exato. Em UIC, cada indutor conserva sua corrente inicial. Rust recebe as duas correntes do par acoplado e contabiliza E=½L1·i1²+M·i1·i2+½L2·i2² desde t=0.

Para uma CCCS de ganho g, a estampa A[p,ctrl]+=g dá ∂V/∂g=−(y_p−y_n)·Ictrl. Para uma CCVS de transresistência r, A[aux,ctrl]+=r dá ∂V/∂r=−y_aux·Ictrl. A corrente de controle é a incógnita auxiliar da fonte de tensão de controle, com o sinal da MNA. Essas derivadas participam do RSS, Pareto e seleção de cantos quando há tolerância; o padrão das fontes dependentes é tolerância zero.
