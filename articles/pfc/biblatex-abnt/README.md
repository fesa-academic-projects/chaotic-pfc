# biblatex-abnt vendorizado

Estes 19 arquivos são o **biblatex-abnt 4.0** (2024-07-04), de Daniel Ballester
Marques e Lucas Sales, copiados **sem nenhuma alteração** do CTAN:

* <https://ctan.org/pkg/biblatex-abnt>
* <https://github.com/abntex/biblatex-abnt>

Eles estão no repositório para o estilo ficar **preso à versão que foi
conferida**: o `latexmkrc` da raiz põe esta pasta no `TEXINPUTS`, então o
XeLaTeX carrega `biblatex-abnt/abnt.bbx` em vez do que vem instalado no TeX
Live (que pode mudar de uma imagem para outra).

**Não edite estes arquivos.**  Isso quebraria a comparação com o upstream e
exigiria renomear os arquivos por causa da licença (LPPL 1.3c).  Os ajustes
para a NBR 6023:2025 estão em `pfccefsa.cls`, na seção "7.1 ESTILO ABNT
VENDORIZADO" — ver [`../docs/vendorizacao-biblatex-abnt.md`](../docs/vendorizacao-biblatex-abnt.md).

## Procedência (sha256)

| arquivo | sha256 |
|---|---|
| `abnt.bbx` | `8fdc0b814216cc8f0eec880a9ebceeb46cc39f082f3314598b1141880d5887d3` |
| `abnt.cbx` | `13c09cf42096ce1852fcd6acc105b06e9eb2979a1bf9cfba6669590df9e49b88` |
| `abnt-numeric.bbx` | `151130bdb1fd5dd529f406a71a5581a52194ac6ad35506ec5d36875d0796568e` |
| `abnt-numeric.cbx` | `13c8aaba825682b01b50f986eda42d6c79a6f976f0dbbf1dbebd3d79481d9a6b` |
| `brazilian-abnt.lbx` | `c4c0c3f2fa161d6bdd91f1f2bcca5940fb1e152802385deba1a5b817de848592` |

Os demais (`abnt-ibid.*`, `abnt-numeric.*` e os 13 `.lbx`) vêm do mesmo
pacote.  Para conferir tudo de uma vez:

```bash
sha256sum biblatex-abnt/* > /tmp/vendorizado.txt   # compare com o zip do CTAN
```

O `\ProvidesFile` do upstream ainda diz "2018-11-17 v3.4" (o mantenedor não
atualizou a string na versão 4.0); o `CHANGELOG.md` do pacote é que registra a
4.0, de 2024-07-04.

## Licença

LPPL 1.3c (ver o cabeçalho de cada arquivo).  A redistribuição dos arquivos
**sem modificação** é permitida; é por isso que os ajustes do repositório ficam
em arquivo próprio (`pfccefsa.cls`) e não dentro destes.

**Estes arquivos não são cobertos pela licença MIT deste repositório** — eles
continuam sob a LPPL 1.3c, do upstream.  Veja
[`../LICENCIAMENTO.md`](../LICENCIAMENTO.md).

## Como atualizar

```bash
# 1. baixe o zip do CTAN e substitua os arquivos desta pasta
# 2. rode as duas conferências
make conferir-norma      # NBR 6023:2025 continua valendo no PDF de teste
make verificar           # fidelidade do template ao modelo oficial
# 3. revise a seção 7.1 do pfccefsa.cls: se o upstream mudar algum driver ou
#    formato que usamos, o ajuste precisa ser revisto
```
