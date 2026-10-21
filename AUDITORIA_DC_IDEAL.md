# Auditoria dos casos aleatórios com indutores ideais em DC

O modelo impõe Vp−Vn=0 por enrolamento, com corrente como incógnita da MNA. Foram examinadas as 60 sementes sem transformadores e as 60 com transformadores.

Os dez casos anteriormente ignorados não têm ponto de operação DC: há uma fonte não nula imposta sobre um caminho de curtos ideais. O modelo exato também identifica três casos adicionais com corrente circulante indeterminada, que a resistência artificial anterior selecionava arbitrariamente.

| Transformadores | Semente | Diagnóstico | rank(A) | rank([A,z]) | Dimensão DC |
|---|---:|---|---:|---:|---:|
| False | 16 | Sem solução DC | 10 | 11 | 11 |
| False | 57 | Sem solução DC | 9 | 10 | 10 |
| True | 8 | Sem solução DC | 8 | 9 | 10 |
| True | 20 | Sem solução DC | 12 | 13 | 14 |
| True | 24 | Correntes não únicas | 12 | 12 | 14 |
| True | 26 | Correntes não únicas | 9 | 9 | 10 |
| True | 27 | Sem solução DC | 11 | 12 | 12 |
| True | 31 | Sem solução DC | 6 | 7 | 7 |
| True | 32 | Sem solução DC | 6 | 7 | 8 |
| True | 43 | Sem solução DC | 6 | 7 | 8 |
| True | 47 | Sem solução DC | 11 | 12 | 12 |
| True | 51 | Correntes não únicas | 8 | 8 | 9 |
| True | 57 | Sem solução DC | 6 | 7 | 8 |

## Evidência e interpretação

- Dez casos têm rank(A)<rank([A,z]): as equações DC são incompatíveis. Três têm rank(A)=rank([A,z])<N: existe uma família de soluções.
- Os ranks são calculados numericamente após escala de linhas e colunas. Uma verificação topológica independente nos testes confirma o caminho de curtos que contradiz a fonte ou o ciclo de enrolamentos que permite corrente livre.
- Os testes das 120 sementes verificam concordância entre solvers para os 107 circuitos com solução única e rejeição explícita nos 13 casos inválidos/indeterminados. Passar na rejeição não é obter uma solução física.
- Sem escolher resistência parasita, condições iniciais ou outro modelo físico, não há uma corrente DC única a reportar nos três ciclos.
- `uic=True` permite fornecer condições iniciais para um transitório bem definido; não cria um ponto DC que não existe.

Reproduzir: `.venv/bin/python -m pytest tests/test_random_dc_cross.py tests/test_ideal_dc_inductors.py -q -rs`.
