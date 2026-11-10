# Proposta de pilares de segurança (v4)

O PDF do roteiro não foi localizado em `articles/pfc/revisao/seguranca/`; a
proposta usa os cinco pilares nomeados no pedido: Confidencialidade (C),
Integridade (I), Disponibilidade (D), Autenticidade (A) e Não repúdio (N).
Nada foi editado no artigo.

## (a) Quadro 1 com a coluna Pilar

| Verificação | O que impede | Obrigatória | Pilar |
|---|---|---|---|
| Estilo e tipos (2) | código fora do padrão e tipos incompatíveis | Sim | I |
| Testes e contratos (3) | regressão funcional e violação de invariantes | Sim | I |
| Regressão dos resultados (1) | alteração não intencional dos arquivos de resultados | Sim | I |
| Execução sem compilador (1) | falha do caminho interpretado | Sim | D |
| Avisos numéricos (1) | valores não numéricos e avisos ignorados | Sim | I |
| Análise estática e segurança (3) | código inseguro, segredos expostos e fluxos inseguros | Sim | C, I |
| Auditoria e compilação do manuscrito (2) | número sem origem e documento que não compila | Sim | A, I |
| Auditoria de dependências (1) | dependências com vulnerabilidades conhecidas | Não | D, I |

Custo: uma coluna a mais; com os largos atuais, **0–2 linhas** de altura extra.

## (b) Frases propostas para a 3.5

> A integridade dos arquivos de resultados e a leitura sem execução de código
> atendem à integridade (I); a validação de entradas, a falha segura e os testes
> de contrato, propriedade e regressão também protegem I e a disponibilidade (D)
> dos artefatos; a detecção de segredos no CI atende à confidencialidade (C); e a
> regra de commits assinados e o DOI do software dão suporte à autenticidade (A)
> e ao não repúdio (N).

> Os dados são públicos e o enlace não tem garantia criptográfica
> (Seção~\ref{sub:limitacoes_pfc}); C, A e N não se aplicam ao sinal, e a
> assinatura dos commits é uma regra do repositório, não uma propriedade do
> resultado científico.

Custo: **cerca de 6 linhas**.

## (c) Trabalhos futuros e pilares

Itens já presentes no texto, sem novidade:

| Item futuro já mencionado | Pilar |
|---|---|
| avaliar o enlace em canais com ruído e medir a taxa de erro de bit | I, D |
| estender a verificação a filtros de resposta ao impulso infinita | I |
| explorar outras larguras de banda e parâmetros do mapa | I |
| implementar gerador e receptor em hardware, para o TRL 4 | D, A |

Custo: **cerca de 3 linhas**, se resumido em uma frase.

## (d) Tabela de mecanismos

| Mecanismo (nome no artigo) | Pilar | Evidência | Limitação | Status |
|---|---|---|---|---|
| verificação de integridade dos arquivos de resultados | I | `src/chaotic_pfc/analysis/sweep/_io.py:147`; `_trusted_checkpoints.py` | cobre os arquivos versionados, não toda entrada externa | implementado |
| leitura sem desserialização de objetos | I | `src/chaotic_pfc/analysis/sweep/_io.py:150` | exige `trust_legacy_pickle` para arquivos antigos | implementado |
| validação de entradas e exceções de domínio | I | `src/chaotic_pfc/_validate.py:38`; `src/chaotic_pfc/errors.py:10` | valida a API, não o conteúdo científico | implementado |
| falha segura | I, D | `src/chaotic_pfc/analysis/sweep/_io.py:147`; `src/chaotic_pfc/_validate.py:28` | interrompe o processamento; não recupera dados perdidos | implementado |
| testes de contrato, propriedade e regressão | I | artigo 3.4; `tests/` | cobrem o que foi testado, não todos os usos | implementado |
| metadados por varredura | I, A | `src/chaotic_pfc/analysis/sweep/_io.py:77`; artigo 3.4 | metadados não são assinados individualmente | implementado |
| detecção de segredos no CI | C | `.github/workflows/ci.yml:953-979`; `.github/rulesets/main.json:80` | protege o repositório, não o sinal transmitido | implementado |
| commits assinados e regras do repositório | A, N | `.github/rulesets/main.json:24` | a regra existe, mas os commits desta entrega não são assinados | regra implementada; entrega não assinada |
| DOI do software | A | artigo, citação `chaoticPFC2026` | identifica a versão, não garante integridade criptográfica | implementado |
| auditoria de dependências | D, I | Quadro 1; CI | é consultiva, não bloqueante | implementado como consulta |
| enlace sem garantia criptográfica | C, A, N | artigo 4.7, “não oferece garantia criptográfica” | não há análise de segurança do esquema | não tratado; limitação |
| dados públicos | C | resultados de simulação, sem dados pessoais | não há confidencialidade a proteger | não se aplica |

## Custo total

(a) 0–2 linhas + (b) 6 linhas + (c) 3 linhas = **9–11 linhas**. Como a última
página tem 5 linhas livres, a aplicação exige enxugar de 4 a 6 linhas de prosa
não solicitada nesta rodada; a proposta não edita o artigo.
