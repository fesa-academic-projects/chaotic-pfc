# Mecanismos por pilar de segurança

Nomes dos mecanismos como ficaram no artigo da v4. Para uso nos slides do grupo.

| Mecanismo (nome no artigo) | Pilar | Evidência | Limitação | Status |
|---|---|---|---|---|
| verificação de integridade dos arquivos de resultados | I | `src/chaotic_pfc/analysis/sweep/_io.py:147`; `_trusted_checkpoints.py` | cobre os arquivos com referência embutida no pacote, não toda entrada externa | implementado |
| leitura sem execução de código | I | `src/chaotic_pfc/analysis/sweep/_io.py:150` | arquivos de versões antigas só são aceitos após a verificação de integridade | implementado |
| validação de entradas | I | `src/chaotic_pfc/_validate.py:38`; `src/chaotic_pfc/errors.py:10` | valida a fronteira da API, não o conteúdo científico | implementado |
| falha segura | D, I | `src/chaotic_pfc/_validate.py:28`; `src/chaotic_pfc/analysis/sweep/_io.py:147` | interrompe o processamento; não recupera dados perdidos | implementado |
| testes de contrato, propriedades e regressão | I | artigo, Seção 3.4; `tests/` | cobrem o que foi testado, não todos os usos | implementado |
| execução sem compilador | D | artigo, Quadro 1 | cobre o caminho interpretado | implementado |
| detecção de segredos expostos | C | `.github/workflows/ci.yml:953-979` | protege o repositório, não o sinal transmitido | implementado |
| regra de commits assinados | A, N | `.github/rulesets/main.json:24` | a regra existe, mas os commits desta entrega não são assinados | regra implementada; entrega não assinada |
| metadados gravados em cada varredura | I, A | `src/chaotic_pfc/analysis/sweep/_io.py:77`; artigo, Seção 3.4 | não são assinados individualmente | implementado |
| auditoria de dependências | D, I | artigo, Quadro 1; CI | é consultiva, não bloqueante | implementado como consulta |
| análise de segurança do esquema de comunicação | C, A | artigo, Seção 5 (trabalhos futuros) e Seção 4.7 | ainda não realizada | futuro |
| enlace sem garantia criptográfica | C, A, N | artigo, Seção 4.7 | não há análise de segurança do esquema | não tratado; limitação |

## Observações

- C = confidencialidade, I = integridade, D = disponibilidade,
  A = autenticidade, N = não repúdio.
- Os dados de simulação não são pessoais nem sigilosos; não há confidencialidade
  a proteger nos resultados. O artigo não trata desse ponto e ele não entra nos
  slides como mecanismo.
- A proposta anterior, com a justificativa de cada linha, está em
  `proposta-pilares-v4.md`.
