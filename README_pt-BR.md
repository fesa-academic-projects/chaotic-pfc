# chaotic-pfc

<p align="right">
  <strong>🇧🇷 Português (Brasil)</strong> |
  <strong><a href="./README.md">🇺🇸 English</a></strong>
</p>

[![CI](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/ci.yml/badge.svg)](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/chaotic-pfc/badge/?version=latest)](https://chaotic-pfc.readthedocs.io/)
[![codecov](https://codecov.io/gh/fesa-academic-projects/chaotic-pfc/branch/main/graph/badge.svg)](https://codecov.io/gh/fesa-academic-projects/chaotic-pfc)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy: checked](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![CodeQL](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/fesa-academic-projects/chaotic-pfc/actions/workflows/github-code-scanning/codeql)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Sistema de comunicação digital caótico baseado no mapa de Hénon.
Projeto de PFC (*Projeto Final de Curso*).

<p align="center">
  <img src="figures/sweeps/Hamming (lowpass)/fig2_classification_interleaved.svg" width="600" alt="Mapa de classificação de Lyapunov — Hamming lowpass">
</p>
<p align="center">
  <em>Classificação dos expoentes de Lyapunov por ordem e cutoff — periódico (azul), caótico (vermelho), divergente (cinza).</em>
</p>

## Sobre o projeto

Este repositório contém a implementação de um sistema de comunicação digital
caótico baseado no mapa de Hénon. O transmissor modula uma mensagem binária
no estado de um oscilador caótico; o receptor recupera a mensagem por
sincronização de Pecora-Carroll. O projeto inclui ainda um estudo numérico
completo dos expoentes de Lyapunov do mapa de Hénon e de variantes de ordem
superior com filtros FIR internos, incluindo varreduras de parâmetros sobre
ordem do filtro e frequência de corte — o que permite classificar regiões do
espaço de parâmetros como caóticas, periódicas ou divergentes (figura acima).

O projeto é organizado como um pacote Python instalável (`chaotic_pfc`)
com uma CLI unificada que reproduz cada experimento. A varredura pesada de
Lyapunov é compilada em JIT com Numba e paralelizada sobre a grade
`(ordem, cutoff)`.

## Instalação

Requer Python 3.11 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,fast]"   # ferramentas de desenvolvimento + Numba
```

Para instalar sem Numba (mais lento, mas sem dependências binárias):

```bash
pip install -e ".[dev]"
```

Para uma reprodução byte-exata do ambiente testado no CI (útil ao
investigar regressões), instale a partir do lock file:

```bash
pip install -r requirements-lock.txt
```

## Uso

Execute o pipeline completo, salvando todas as figuras:

```bash
chaotic-pfc run all --no-display
```

Execute um experimento individual:

```bash
chaotic-pfc run attractors --save
chaotic-pfc run lyapunov --save --n-ci 20
```

Execute a varredura de Lyapunov. A varredura completa é pesada (dezenas de
minutos a horas, dependendo do hardware); o modo rápido usa uma grade reduzida
para testar o pipeline em segundos:

```bash
# Varredura completa para uma combinação (janela, filtro)
chaotic-pfc run sweep compute --window hamming --filter lowpass

# Teste rápido (~segundos)
chaotic-pfc run sweep compute --window hamming --filter lowpass --quick

# Gera as 4 figuras de classificação (PNG + SVG) a partir do .npz salvo
chaotic-pfc run sweep plot --window hamming --filter lowpass
```

Os checkpoints em `data/sweeps/*.npz` são versionados no repositório, então
os gráficos podem ser regenerados a qualquer momento sem reexecutar a varredura.

Explore todos os comandos com `chaotic-pfc run --help` e as flags de cada
subcomando com `chaotic-pfc run <nome> --help`.

## Estrutura do projeto

```
chaotic-pfc/
├── pyproject.toml                 Metadados e dependências do pacote
├── src/chaotic_pfc/               Biblioteca instalável
│   ├── dynamics/                  Mapas de Hénon, expoentes de Lyapunov, sinais, análise espectral
│   ├── comms/                     Transmissor, modelos de canal, receptor, esquemas DCSK
│   ├── analysis/                  Varreduras de parâmetros, pós-processamento estatístico, gráficos
│   │   └── sweep/                 Kernel JIT com Numba, pré-computação FIR, E/S
│   ├── plotting/                  Figuras com qualidade de publicação (atratores, sensibilidade, comunicação)
│   ├── cli/                       Módulos da CLI unificada
│   └── config.py                  Configuração centralizada
├── tests/                         Testes unitários
├── data/
│   ├── lyapunov/                  Tabelas CSV do protocolo ensemble
│   └── sweeps/                    Checkpoints .npz versionados das varreduras
├── figures/                       Figuras finais (SVG para o artigo, PNG para preview)
└── scripts/
    └── benchmark.py               Benchmarks de performance para operações principais
```

### API pública

O namespace `chaotic_pfc` reexporta ~60 símbolos que formam a API pública
estável. Eles são importáveis diretamente de `chaotic_pfc`:

```python
from chaotic_pfc import run_sweep, henon_standard, fir_channel, SweepResult
```

Detalhes internos de implementação (módulos privados com prefixo underscore, ex.
`chaotic_pfc.analysis.sweep._kernel`) podem mudar sem aviso e devem ser
importados apenas em testes ou scripts de pesquisa avançada.

Extras opcionais seguem um caminho de import separado. Por exemplo, a
visualização 3-D requer o extra `viz3d` (`pip install -e '.[viz3d]'`)
e é importada diretamente do seu módulo:

```python
from chaotic_pfc.analysis.sweep_plotting_3d import plot_3d_beta_volume
```

## Experimentos

| Subcomando | Descrição |
|------------|-----------|
| `chaotic-pfc run attractors`         | Retratos no espaço de fase das três variantes de Hénon. |
| `chaotic-pfc run sensitivity`        | Sensibilidade às condições iniciais (SDIC). |
| `chaotic-pfc run comm-ideal`         | Transmissor e receptor sobre um canal ideal. |
| `chaotic-pfc run comm-fir`           | Sistema ponta a ponta com canal FIR. |
| `chaotic-pfc run comm-order-n`       | Comunicação com mapa de ordem N filtrado. |
| `chaotic-pfc run lyapunov`           | Espectros de Lyapunov: CI única e ensemble para sistemas 2-D e 4-D. |
| `chaotic-pfc run sweep compute`      | Varredura paralela de Lyapunov sobre `(ordem × cutoff)`. |
| `chaotic-pfc run sweep plot`         | Mapas de classificação a partir dos dados salvos. |
| `chaotic-pfc run all`                | Pipeline completo, em ordem. |

## Testes

Execute a suíte completa:

```bash
pytest
```

Exclua testes lentos durante o desenvolvimento:

```bash
pytest -m "not slow"
```

Ou use o Makefile:

```bash
make test          # suíte completa
make test-fast     # exclui testes lentos
make check-all     # lint + format + typecheck + testes rápidos
```

## Comandos do Makefile

| Comando | Ação |
|--------|------|
| `make test` | Executa a suíte completa de testes |
| `make test-fast` | Executa testes excluindo `@pytest.mark.slow` |
| `make lint` | Ruff linter |
| `make format` | Ruff auto-format |
| `make format-check` | Verifica formatação sem alterar arquivos |
| `make typecheck` | mypy static type checker |
| `make check-all` | lint + format-check + typecheck + test-fast |
| `make docs` | Compila documentação Sphinx HTML (inglês) |
| `make docs-pt` | Compila documentação Sphinx HTML (português) |
| `make docs-pdf` | Compila documentação Sphinx PDF |
| `make docs-epub` | Compila documentação Sphinx EPUB |
| `make benchmark` | Benchmarks de performance |
| `make pre-commit` | Executa todos os hooks de pre-commit |
| `make clean` | Remove artefatos de build e cache |
| `make help` | Mostra todos os comandos |

## Documentação

A documentação completa está disponível em
[chaotic-pfc.readthedocs.io](https://chaotic-pfc.readthedocs.io/)
em inglês e português (Brasil), com formatos para download: PDF, EPUB e
HTMLZip.

Compile a documentação HTML localmente:

```bash
pip install -e ".[docs]"
cd docs
make html
```

Abra `docs/_build/html/index.html` no navegador. Cada módulo gera
automaticamente uma página de referência da API a partir das docstrings
no estilo NumPy.

## Desenvolvimento

Ative os hooks de pre-commit para que o Ruff execute automaticamente a
cada `git commit`:

```bash
pre-commit install
```

Execute todos os hooks manualmente (útil após `pre-commit autoupdate`
ou antes de push):

```bash
pre-commit run --all-files
```

## Licença

Distribuído sob os termos da Licença BSD 3-Clause. Veja
[LICENSE](LICENSE) para o texto completo.

## Autores

Desenvolvido por alunos da **Faculdade Engenheiro Salvador Arena (FESA)**.
Veja [AUTHORS](AUTHORS) para a lista completa de contribuidores.
