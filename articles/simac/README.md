# VIII SIMAC — registro congelado

## Situação

Trabalho aprovado no **VIII SIMAC** como *trabalho em andamento*. Esta é a
**versão final**; **não será editado**.

**Título:** *Preservação do regime caótico no mapa de Hénon com filtros FIR:
caracterização do espaço de parâmetros por tipo de filtro e janela de projeto*.

Os **PDFs** (`simac_identificado.pdf`, `simac_nao_identificado.pdf`) são o
**registro oficial** — o texto aprovado. Os `.docx`
(`simac_identificado.docx`, `simac_nao_identificado.docx`) foram reconstruídos
em **2026-10-12** a partir dos PDFs, reproduzindo o layout e a paginação, para
permitir eventuais consultas de texto; **não substituem os PDFs** como fonte de
verdade. O PDF não identificado foi originalmente depositado como
`simac__nao_identificado.pdf` (com dois underscores) e foi renomeado para
`simac_nao_identificado.pdf` ao ser versionado aqui.

## Código e dados usados na época

Os números do SIMAC para a janela **Hamming** (pontos caóticos `Nc` / fração
dos pontos de órbita limitada `rho_fin` / expoente médio `λ̄`) correspondem
exatamente ao estado do repositório no commit **`f517e0a`** (*"chore(data):
purge sentinel -1e30 from infected Lyapunov ensembles"*, 2026-05-23):

| Cenário | Nc | rho_fin (%) | λ̄ |
|---|---|---|---|
| **SIMAC (PDF)** | passa-altas 72 · passa-faixa 294 · rejeita-faixa 55 | 3,8 · 7,9 · 35,5 | 0,0234 · 0,0273 · 0,0874 |
| **f517e0a** (match exato) | 72 · 294 · 55 | 3.8 · 7.9 · 35.5 | 0.0234 · 0.0273 · 0.0874 |
| `ac34be3` (2026-08-16) | 72 · 294 · 55 | 3.8 · 7.9 · 35.5 | **0.0233** · 0.0273 · **0.0873** |
| atual (regeneração 0.9.0 com o ponto fixo sem cancelamento, 2026-10-11) | 72 · **295** · 55 | 3.8 · 7.9 · 35.5 | 0.0233 · **0.0276** · **0.0870** |

Reprodução da linha de `f517e0a`:

```bash
git archive f517e0a data/sweeps | tar -x -C /tmp/simac
python tools/paper_numbers.py --sweeps /tmp/simac/data/sweeps --json /tmp/simac.json
# T3.rank21 = highpass | Hamming | Nc=72  | rho_fin=3.8 | mean=0.0234
# T3.rank10 = bandpass | Hamming | Nc=294 | rho_fin=7.9 | mean=0.0273
# T3.rank23 = bandstop | Hamming | Nc=55  | rho_fin=35.5 | mean=0.0874
```

As 124 varreduras foram depois regeneradas com o ponto fixo racionalizado
(regeneração 0.9.0 com o ponto fixo sem cancelamento), de modo que os `.npz` versionados **não** reproduzem os números do
SIMAC — daí o `ac34be3` (meio caminho) e o estado atual divergirem nos últimos
dígitos.

As figuras podem ser reproduzidas com
`chaotic-pfc run paper-figures --only simac` (`src/chaotic_pfc/cli/paper_figures.py`),
observando que elas usam os `.npz` **atuais**, não os de `f517e0a`.

Nada em `tools/`, no `Makefile`, no `check-paper` ou no CI processa
`articles/simac/` — é um registro congelado, sem gerador nem verificação.

## Errata

Sem alterar o `.docx` nem os PDFs (que permanecem como aprovados):

1. **Ganho em frequência zero.** O parágrafo afirma que os filtros com ganho
   próximo de zero (passa-altas e passa-faixa) “deslocam o ponto de equilíbrio
   positivo … para regiões em que a órbita tende a divergir”. A análise
   posterior (`tools/paper_numbers.py`, rótulos `GAIN.*`) mostra que
   **99,9 %** dos pontos com `|G| < 0,1` são periódicos e apenas **27** são
   caóticos; nenhum dos 2 900 pontos caóticos do passa-altas tem `|G| < 0,1`.
2. **Título da referência BORGES; EISENCRAFT (2022).** No SIMAC consta *"Um
   estudo numérico do exponente de Lyapunov do Mapa Hénon com filtro FIR"*; o
   correto é *"**Um Estudo Numérico do Expoente de Lyapunov do Mapa de Hénon com
   Filtro FIR**"* (`articles/jcis/references.bib`, entrada `Borges2022sbrt`).
3. **Números da janela Hamming.** Com os dados atuais, os valores passam a
   **passa-altas 72 / 3,8 % / 0,0233**, **passa-faixa 295 / 7,9 % / 0,0276** e
   **rejeita-faixa 55 / 35,5 % / 0,0870**.
