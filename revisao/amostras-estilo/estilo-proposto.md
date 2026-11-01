# Estilo único de figuras — proposta para aprovação

Escopo: as figuras dos dois artigos (PFC e JCIS). Nada foi aplicado nos
manuscritos; as amostras desta pasta são geradas por `tools/style_samples.py`
com o módulo `tools/paper_style.py`.

## 1. De onde veio cada escolha

Não havia nenhum PDF em `articles/referencias-estilo/` (a pasta não existe),
então usei o repositório do orientador e as figuras do JCIS que ele aprovou.

| Fonte | O que foi lido | O que saiu |
|---|---|---|
| `github.com/vinicius-s-borges/Chaotic-properties-of-an-FIR-filtered-Henon-map` | `figure_1.m`, `_2`, `_3`, `_5`, `_7`, `_9`, `_10`, `_12`, `_14` (inteiros); `_4`, `_8`, `_11`, `_13`, `_15` (busca por `plot`, `pcolor`, `set(gca)`, `LineWidth`, `FontSize`) | `LineWidth` 2 nas curvas e nos eixos, `FontSize` 20–24 nos ticks, `MarkerSize` 1 nos pontos de órbita, `grid on`, `layer=top`, azul sólido + preto, preto tracejado como referência, vermelho tracejado nos guias de projeto, LaTeX no eixo |
| JCIS aprovado — `articles/jcis/Figures/*classification*.svg` | cores das quatro figuras de classificação | caótico `#FF0000`, não caótico `#4DBEEE`, divergente `#E0E0E0`, borda cinza `#808080` |
| JCIS aprovado — Fig. 1 e Fig. 3 (`article.pdf`) | página 2 e página 8 | Fig. 1 azul contínuo + vermelho tracejado; Fig. 3 legenda dentro do painel (a), eixos 0–1 com passo 0,2 |
| `pfccefsa.cls` | seção 3 (tamanhos e entrelinhas) | texto 12 bp, legenda 11 bp (`\tamonze`), área de texto 510,22 bp = 7,086 in |
| `IEEEtran.cls` | `\@makecaption` (journal) | legenda em `\footnotesize` = 8 pt, coluna 3,5 in |
| `articles/pfc/tools/gerar_figuras_pfc.py`, `gerar_psd_canal.py`, `make_hamming_lowpass_map.py`, `src/chaotic_pfc/paper/figures.py` | pipeline atual | tamanhos, frações de `\textwidth` e defeitos listados na seção 4 |

Duas correções em relação ao que o pipeline faz hoje:

- A linha do PFC não usa `mathptmx`: o `pfccefsa.cls` compila com XeLaTeX e
  `fontspec`, e o PDF sai em **Liberation Serif** (não há Times New Roman na
  máquina). O JCIS sai em **NimbusRomNo9L** (`mathptmx`).
- Por isso a fonte da figura é a do corpo de cada artigo, não uma só:
  Liberation Serif no PFC e STIXGeneral (desenho Times, TrueType) no JCIS. O
  Nimbus só existe em CFF nesta máquina e teria de ser embutido como Type 3;
  STIX evita isso e embute como TrueType.

## 2. Parâmetros propostos

| Parâmetro | PFC | JCIS |
|---|---|---|
| Área de texto | 7,086 in (510,22 bp) | 3,5 in (coluna IEEEtran) |
| Legenda no impresso | 11 pt (`\tamonze`) | 8 pt (`\footnotesize`) |
| Tamanho de **todo** texto da figura | 11 pt | 8 pt |
| Fonte | Liberation Serif + matemática STIX | STIXGeneral + STIX |
| Geração | já na largura final: `fração × 7,086 in` | já na largura final: 3,5 in |
| Decimal | vírgula, menos tipográfico | ponto |
| Idioma dos rótulos | português | inglês |
| Classificação | caótico vermelho, não caótico azul, divergente cinza claro | idem |
| Séries de linha | azul sólido → vermelho tracejado → preto → cinza tracejado | idem |
| Espessuras | curva 1,0 pt; moldura 0,8 pt; grade 0,4 pt | idem |
| Marcadores | órbita 1,5–3 pt; projeto escolhido: círculo vazado 4 pt | idem |
| Saída | PDF vetorial, fontes Type 42, sem data, página = `figsize` | idem |

Tamanhos: o MATLAB do orientador foi desenhado com `FontSize` 20–24 e
`LineWidth` 2 em figuras de ~2× o tamanho impresso; 11 pt de texto, 1,0 pt de
curva e 0,8 pt de moldura são a conversão para o tamanho final.

## 3. Luminância da classificação (escala de cinza)

`Y` = luminância relativa (WCAG), `L*` = luminosidade CIE, `cinza` = valor
BT.601 de 8 bits, que é o que a conversão para cinza produz.

| Classe | Cor | Y | L* | cinza |
|---|---|---|---|---|
| Caótico | `#FF0000` | 0,213 | 53,2 | 76 |
| Não caótico | `#4DBEEE` | 0,446 | 72,6 | 162 |
| Divergente | `#E0E0E0` | 0,745 | 89,2 | 224 |

Degraus: 19 e 17 em L*; **86 e 62 níveis de cinza** (de 255). Medido nos PNGs
gerados (`pfc-fig9-mapa-cinza.png`, `jcis-fig3-mapa-cinza.png`): 76 / 162 /
224, exatamente os valores da tabela. As três classes continuam separadas em
cinza; a paleta roxo/verde-azulado do PFC tinha 15 / 55 / 87, que também separa,
mas foge da convenção do orientador.

## 4. Figuras afetadas

### PFC (14 figuras)

| Fig. | Arquivo | Situação hoje | Proposta |
|---|---|---|---|
| 1 | `fig_henon_signals_h` | 10 × 3 in inserida em `\textwidth` (escala 0,71); texto de 3,2 pt a 10,2 pt; azul (`tab:blue`) × laranja (`tab:orange`) | 7,09 × 2,13 in, 11 pt, azul sólido × vermelho tracejado |
| 2 | `fig_fir_respostas` | 10 × 3,4 in em `\textwidth`; texto de 1,6 pt a 10,2 pt; quatro séries no ciclo padrão | 7,09 in, 11 pt, ordem do orientador |
| 3 | TikZ (gerador) | texto na fonte do corpo | conferir apenas |
| 4 | TikZ (enlace sem filtro) | idem | conferir apenas |
| 5 | TikZ (arquitetura) | `rotulo` em `\footnotesize` × `scale=0.8` = 8,0 bp | `\tamonze` (11 bp), sem redução |
| 6 | TikZ (estimador) | `\footnotesize` × `scale=0.88` = 8,8 bp | idem |
| 7 | TikZ (enlace filtrado) | texto na fonte do corpo | conferir apenas |
| 8 | `fig_comm_falha` | 5,11 in inserida a 0,82 (`\textwidth`) = escala 1,14; texto de 7,6 pt; azul | 5,81 in, 11 pt |
| 9 | quatro mapas de classificação | roxo escuro / verde-azulado / cinza; texto de 7,1 a 8,8 pt; legenda dentro de (a) | vermelho / azul / cinza claro; 11 pt; legenda única acima dos quatro painéis |
| 10 | `fig_chaotic_density_norm` | 0,9 `\textwidth` = escala 1,09; texto de 6,8 pt; paleta roxo/verde | 6,38 in, 11 pt, paleta nova |
| 11 | `fig_kaiser_beta` | 0,8 `\textwidth`; texto de 7,9 pt; quatro séries no ciclo padrão (azul, laranja, verde, vermelho) | 5,67 in, 11 pt, ordem do orientador |
| 12 | `hamming_lowpass_classification` | rótulo "case" sobrando; círculo invisível; decimais com ponto; texto de 3,5 a 7,8 pt | sem "case"; círculo vazado de 4 pt com borda preta; vírgula; 11 pt |
| 13 | `fig_comm_filtro` | 0,8 `\textwidth`; texto de 7,6 pt; azul | 5,67 in, 11 pt |
| 14 | `fig_psd_laco` | decimais com ponto; laranja (`#d55e00`); texto de 8,7 a 11,6 pt | vírgula; vermelho tracejado; 11 pt; "limite do canal" passa para a legenda |

Menor texto efetivo hoje, contra 11 pt de legenda: Fig. 2 = **1,6 pt**, Fig. 1
= **3,2 pt**, Fig. 12 = **3,5 pt**, Fig. 10 = 6,8 pt, Fig. 9 = 7,1 pt, Figs.
8 e 13 = 7,6 pt, Fig. 11 = 7,9 pt, Fig. 14 = 8,7 pt. Nenhuma figura do PFC
cumpre a regra hoje.

### JCIS (6 figuras)

| Fig. | Situação | Proposta |
|---|---|---|
| 1 | azul sólido × vermelho tracejado, 9/8/7,5 pt em 3,5 in | mantém; fonte passa a STIXGeneral; 8/8/8 pt |
| 2 | TikZ (gerador) | conferir apenas |
| 3 | 2 × 2 mapas aprovados, vermelho/azul/cinza | mantém cores; 8 pt em todo o texto; legenda unificada na ordem caótico → não caótico → divergente |
| 4 | `fig_chaotic_density_norm` | paleta nova, 8 pt |
| 5 | `fig_kaiser_beta` | ordem de séries do orientador, 8 pt |
| 6 | `fig_psd` | ponto decimal (mantém), 8 pt, preto × vermelho tracejado |

As figuras do JCIS já são geradas na largura final e usam 9/8/7,5 pt: só os
rótulos de eixo (9 pt) ficam 1 pt acima da legenda; a unificação leva tudo a
8 pt.

## 5. Mudanças que caem no texto

- Legenda da Fig. 9 do PFC: "Caótico em roxo escuro, não caótico em verde
  azulado e divergente em cinza claro" passa a "caótico em vermelho, não
  caótico em azul e divergente em cinza claro".
- Nenhuma largura de `\includegraphics` muda: a figura nasce no tamanho em que
  é inserida.
- Fig. 12: a legenda já descreve o círculo; o rótulo "case" sai.

## 6. Como aplicar (depois da aprovação)

1. `tools/paper_style.py` (pronto) vira a única fonte de estilo.
2. Os quatro geradores passam a chamar `apply("pfc"|"jcis")` e
   `install_sweep_palette()`; os tamanhos viram `st.caption_pt` e as larguras
   `fração × st.textwidth_in`.
3. Figuras 1 e 2 do PFC não têm gerador no repositório: entram no pipeline
   (a Fig. 1 é a mesma rotina da Fig. 1 do JCIS, com vírgula e 11 pt).
4. Figs. 5 e 6 do PFC: `rotulo` para `\tamonze` e remover a redução por
   `scale`/`transform shape`.
5. Regerar, recompilar os dois artigos, rodar as auditorias de números e
   páginas, e conferir a prova em cinza.

## 7. Decisões pendentes

1. Fig. 14: o rótulo vertical "limite do canal" não cabe em 11 pt; a amostra
   o move para a legenda (a legenda da figura já cita a linha pontilhada).
2. Ordem da legenda de classificação: unificar em caótico → não caótico →
   divergente (hoje o JCIS começa pelo não caótico).
3. Fig. 1: manter a proporção atual (altura 2,13 in) ou aceitar 2,4 in para
   dar folga à legenda de duas entradas em 11 pt.
