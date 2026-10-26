# Procedência do modelo PFC

## Origem

- Upstream: https://gitlab.com/ecfesa/pfc-template
- Commit: `597c54f597c00a13962f54dd3a7f8df370fafd48`
- Pacote recebido como `pfc-template-main.zip`; o comentário do arquivo
  registra o mesmo commit.
- O modelo oficial é a classe `pfccefsa`, versão V-05, da Faculdade
  Engenheiro Salvador Arena.

## Arquivos copiados

- `pfccefsa.cls`
- `biblatex-abnt/` (diretório completo)
- `latexmkrc`
- `image1.png`
- `image2.png`
- `LICENSE`
- `tools/conferir_norma.py`
- `tools/conferir_citacoes.py`

`articles/pfc/artigo.tex` foi criado a partir de `template_artigo.tex`.
Foram removidos o exemplo de figura que dependia de `figura9.pdf` e a
menção a esse arquivo. Travessões tipográficos em comentários foram
substituídos por dois-pontos ou hífen, sem alteração de conteúdo técnico.
O mesmo ajuste tipográfico foi feito nos comentários de `latexmkrc` e das
duas ferramentas copiadas. As ferramentas também receberam docstrings e uma
marca de lint, sem mudança funcional.

## Integridade

`pfccefsa.cls` e `biblatex-abnt/` estão sem modificação em relação ao
upstream. Nenhum arquivo de `referencia/`, `exemplo_artigo.tex`,
`figura9.pdf`, `teste_abnt2025.tex`, `teste_abnt2025.bib` ou da CI do
GitLab foi copiado.

## Licenças

- `pfccefsa.cls`, `latexmkrc`, `tools/conferir_norma.py` e
  `tools/conferir_citacoes.py`: MIT, conforme `LICENSE`.
- `biblatex-abnt/`: LPPL 1.3c, de Daniel Ballester Marques e Lucas Sales.
  Os arquivos são mantidos sem alteração e não são relicenciados pela MIT.
- `image1.png` e `image2.png`: logotipos e identidade visual da faculdade.
  São marcas institucionais e não ficam sob a licença do repositório.

## Compilação

O alvo `make pfc-pdf` compila `articles/pfc/artigo.tex` com
`latexmk -pdfxe`, usando `biber` quando o documento tiver referências
BibLaTeX. O ambiente de verificação usa TeX Live 2026, XeLaTeX 3.141592653
e biber 2.22; a classe resolve a fonte para Liberation Serif.
