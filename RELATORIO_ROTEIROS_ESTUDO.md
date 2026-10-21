# Validação dos cinco roteiros de estudo

Executado com CPython 3.12.14, NumPy 2.2.6 e SciPy 1.14.1, com Rust disponível.

Todas as verificações abaixo passaram. O relatório só é escrito após todas as verificações.

| Roteiro | Configuração | Valores observados |
|---|---|---|
| Divisor | dense | Vout (V): 7.2; I (A): 1.2; P fornecida (W): 14.4; resíduo potência (W): -2.66453526e-15 |
| Divisor | sparse | Vout (V): 7.2; I (A): 1.2; P fornecida (W): 14.4; resíduo potência (W): -2.66453526e-15 |
| Divisor | rust_sparse | Vout (V): 7.2; I (A): 1.2; P fornecida (W): 14.4; resíduo potência (W): -2.66453526e-15 |
| RC | euler/python | ordem observada: 0.988929714; erro dt=tau/80 (V ou A): 0.0228734559; resíduo energia (J): 2.79667004e-07 |
| RC | euler/rust | ordem observada: 0.988929714; erro dt=tau/80 (V ou A): 0.0228734559; resíduo energia (J): 2.79667004e-07 |
| RC | trapezoidal/python | ordem observada: 2.0001832; erro dt=tau/80 (V ou A): 4.79017798e-05; resíduo energia (J): -1.94828407e-09 |
| RC | trapezoidal/rust | ordem observada: 2.0001832; erro dt=tau/80 (V ou A): 4.79017797e-05; resíduo energia (J): -1.94828407e-09 |
| RL | euler/python | ordem observada: 0.988929714; erro dt=tau/80 (V ou A): 0.00228734559; resíduo energia (J): 2.79667004e-05 |
| RL | euler/rust | ordem observada: 0.988929714; erro dt=tau/80 (V ou A): 0.00228734559; resíduo energia (J): 2.79667004e-05 |
| RL | trapezoidal/python | ordem observada: 2.0001832; erro dt=tau/80 (V ou A): 4.79017797e-06; resíduo energia (J): -1.94828407e-07 |
| RL | trapezoidal/rust | ordem observada: 2.0001832; erro dt=tau/80 (V ou A): 4.79017797e-06; resíduo energia (J): -1.94828407e-07 |
| Transformador | k=0.5/python | Vsec RMS (V): 4.20228772; fase Vsec (graus): -81.947741; E final (J): 0.00587308646; resíduo energia (J): -8.01307586e-10 |
| Transformador | k=0.5/rust | Vsec RMS (V): 4.20228772; fase Vsec (graus): -81.947741; E final (J): 0.00587308646; resíduo energia (J): -8.0130758e-10 |
| Transformador | k=-0.5/python | Vsec RMS (V): 4.20228772; fase Vsec (graus): 98.052259; E final (J): 0.00587308646; resíduo energia (J): -8.01307586e-10 |
| Transformador | k=-0.5/rust | Vsec RMS (V): 4.20228772; fase Vsec (graus): 98.052259; E final (J): 0.00587308646; resíduo energia (J): -8.0130758e-10 |
| Trifásico | ABC | P (W): 1985.5322; Q (var): 748.530328; FP: 0.935714859; corrente neutro (A): 3.61462429e-15 |
| Trifásico | ACB | P (W): 1985.5322; Q (var): 748.530328; FP: 0.935714859; corrente neutro (A): 3.20237283e-15 |
| Tolerâncias | python | RSS linear (V): 0.176776695; sigma linear (V): 0.102062073; sigma quadratura (V): 0.102087608; sigma Monte Carlo (V): 0.102095905; média Monte Carlo (V): 5.00020364 |
| Tolerâncias | rust | RSS linear (V): 0.176776695; sigma linear (V): 0.102062073; sigma quadratura (V): 0.102087608; sigma Monte Carlo (V): 0.102024405; média Monte Carlo (V): 4.99968228 |

## Referências e hipóteses

- Divisor: Vout=12·6/(4+6)=7,2 V; I=1,2 A; P fornecida=14,4 W. `P_resistor` retorna dissipação positiva; `power_audit` apresenta absorção negativa.
- RC/RL: resposta exponencial com tau=1 ms; passos tau/20, tau/40, tau/80; ordem esperada 1 em Euler e 2 no trapézio. O resíduo de energia inclui a integração numérica; Euler tem dissipação numérica.
- Transformador: referência AC por eliminação da corrente secundária; transitório por exponencial matricial da matriz de indutâncias. Energia inclui M·i1·i2.
- Trifásico: cargas desiguais e neutro flutuante; vn=sum(V/Z)/sum(1/Z). A fonte recebe RMS e gera amplitude de pico. Os wattímetros recebem fasores RMS; a soma das potências dos componentes usa os fasores de pico do solver e o fator 1/2.
- Tolerâncias: dois resistores uniformes independentes ±5%, fonte fixa. Cantos exatos deste divisor: [4,75; 5,25] V. RSS não é automaticamente 3σ. A quadratura calcula os momentos do divisor exato; Monte Carlo usa 8.000 amostras e semente 42 em cada backend. Os fluxos aleatórios Python/Rust diferem; a comparação é estatística.

Reproduzir: `.venv/bin/python scripts/validate_study_routes.py`.
