#!/usr/bin/env python3
r"""Confere as citações de um artigo contra o arquivo referencias.bib.

Uso:
    python3 tools/conferir_citacoes.py [artigo.tex ...] [--bib referencias.bib]

Sem argumentos, confere todos os .tex da pasta que usam
\\addbibresource{referencias.bib} (hoje: exemplo_artigo.tex).

O que é conferido
  1. ERRO   - toda chave citada (\\cite, \\citeonline, \\nocite) existe no .bib;
  2. AVISO  - toda entrada do .bib é citada em algum dos .tex (ou registrada
              com \\nocite); entradas não usadas ficam fora da lista gerada;
  3. ERRO   - cada entrada tem os campos obrigatórios do seu tipo (tabela
              OBRIGATORIOS abaixo, mínimo da NBR 6023 / do BibLaTeX);
  4. AVISO  - campos recomendados ausentes (volume, páginas, local, editora…);
  5. AVISO  - "url" sem "urldate" (NBR 6023: data de acesso);
  6. AVISO  - DOI ou título repetido em duas entradas (ficha duplicada).

Sai com código 1 se houver ERRO (dá para usar em CI) e 0 só com avisos.
Não precisa de nenhuma biblioteca externa: só a biblioteca padrão.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Campos obrigatórios e recomendados por tipo de entrada.
# A tupla é (obrigatórios, recomendados); "ano" aceita "year" ou "date".
OBRIGATORIOS: dict[str, tuple[set[str], set[str]]] = {
    "article": ({"author", "title", "journaltitle", "ano"}, {"volume", "pages"}),
    "book": ({"author", "title", "publisher", "ano"}, {"location"}),
    "online": ({"author", "title", "url", "urldate"}, set()),
    "misc": ({"author", "title", "ano"}, {"location", "publisher"}),
    "dataset": ({"author", "title", "ano"}, {"publisher", "doi", "url"}),
    "thesis": ({"author", "title", "type", "institution", "ano"}, {"location", "url"}),
    "inproceedings": ({"author", "title", "booktitle", "ano"}, {"location", "publisher", "pages"}),
    "incollection": ({"author", "title", "booktitle", "publisher", "ano"}, {"location", "pages"}),
}

CITE = re.compile(r"\\(cite|citeonline|textcite|nocite)\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ENTRADA = re.compile(r"@(\w+)\s*\{\s*([^,]+),", re.IGNORECASE)
CAMPO = re.compile(r"^\s*([a-zA-Z]+)\s*=\s*", re.MULTILINE)


def sem_comentarios(texto: str) -> str:
    """Remove comentários LaTeX de *texto*."""
    return re.sub(r"(?<!\\)%.*", "", texto)


def chaves_citadas(tex: str) -> tuple[set[str], set[str]]:
    r"""Devolve (chaves citadas no texto, chaves apenas registradas com \\nocite)."""
    citadas: set[str] = set()
    registradas: set[str] = set()
    for comando, lista in CITE.findall(sem_comentarios(tex)):
        destino = registradas if comando == "nocite" else citadas
        destino.update(k.strip() for k in lista.split(",") if k.strip())
    return citadas, registradas


def ler_bib(bib: str) -> dict[str, dict[str, object]]:
    """Devolve {chave: {'tipo': ..., 'campos': {...}, 'titulo': ...}}."""
    entradas: dict[str, dict[str, object]] = {}
    posicoes = [(m.start(), m.group(1).lower(), m.group(2).strip()) for m in ENTRADA.finditer(bib)]
    for i, (inicio, tipo, chave) in enumerate(posicoes):
        fim = posicoes[i + 1][0] if i + 1 < len(posicoes) else len(bib)
        corpo = bib[inicio:fim]
        # cada campo vai de "nome =" até o começo do próximo campo
        campos: dict[str, str] = {}
        marcas = list(CAMPO.finditer(corpo))
        for j, m in enumerate(marcas):
            valor = corpo[m.end() : marcas[j + 1].start() if j + 1 < len(marcas) else len(corpo)]
            campos[m.group(1).lower()] = valor.strip().rstrip(",").strip().strip("{}").strip()
        entradas[chave] = {"tipo": tipo, "campos": campos}
    return entradas


def presente(campos: dict[str, str], nome: str) -> bool:
    """Informa se *nome* existe em *campos*."""
    if nome == "ano":
        return "year" in campos or "date" in campos
    if nome == "pages":
        return "pages" in campos or "eid" in campos  # nº de artigo substitui páginas
    return nome in campos


def main(argv: list[str] | None = None) -> int:
    """Confere as citações e devolve o código de saída."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "tex", nargs="*", help="arquivos .tex do artigo (padrão: todos os que usam referencias.bib)"
    )
    ap.add_argument(
        "--bib", default="referencias.bib", help="arquivo .bib (padrão: referencias.bib)"
    )
    args = ap.parse_args(argv)

    bib_path = Path(args.bib)
    if not bib_path.exists():
        print(f"ERRO: {bib_path} não existe")
        return 1

    if args.tex:
        tex_paths = [Path(p) for p in args.tex]
    else:
        tex_paths = sorted(
            p
            for p in Path(".").glob("*.tex")
            if f"addbibresource{{{bib_path.name}}}"
            in sem_comentarios(p.read_text(encoding="utf-8"))
        )
    if not tex_paths:
        print(f"ERRO: nenhum .tex usa \\addbibresource{{{bib_path.name}}} (informe os arquivos)")
        return 1

    bib = ler_bib(bib_path.read_text(encoding="utf-8"))
    citadas: set[str] = set()
    registradas: set[str] = set()
    for p in tex_paths:
        c, n = chaves_citadas(p.read_text(encoding="utf-8"))
        citadas |= c
        registradas |= n
        faltando = c - set(bib)
        if faltando:
            print(
                f"ERRO: {p.name} cita chave(s) que não existe(m) em {bib_path}: {', '.join(sorted(faltando))}"
            )

    erros = 0
    avisos: list[str] = []

    for chave, dados in bib.items():
        campos = dados["campos"]  # type: ignore[assignment]
        tipo = dados["tipo"]
        if tipo not in OBRIGATORIOS:
            avisos.append(f"{chave}: tipo @{tipo} sem tabela de campos obrigatórios")
            continue
        obrig, recom = OBRIGATORIOS[tipo]
        falta = sorted(c for c in obrig if not presente(campos, c))
        if falta:
            print(f"ERRO: {chave} (@{tipo}) sem campo(s) obrigatório(s): {', '.join(falta)}")
            erros += 1
        falta_recom = sorted(c for c in recom if not presente(campos, c))
        if falta_recom:
            avisos.append(
                f"{chave} (@{tipo}) sem campo(s) recomendado(s): {', '.join(falta_recom)}"
            )
        if "url" in campos and not presente(campos, "urldate"):
            avisos.append(f"{chave}: url sem urldate (data de acesso)")

    # fichas duplicadas (mesmo DOI ou mesmo título normalizado)
    def chave_dup(campo: str, normalizar=lambda v: v.lower()) -> None:
        vistos: dict[str, str] = {}
        for chave, dados in bib.items():
            valor = dados["campos"].get(campo) if campo in dados["campos"] else None  # type: ignore[union-attr]
            if not valor:
                continue
            v = normalizar(str(valor))
            if v in vistos:
                avisos.append(f"possível duplicata ({campo}): {vistos[v]} e {chave}")
            else:
                vistos[v] = chave

    chave_dup(
        "doi",
        lambda v: (
            v.lower().removeprefix("https://doi.org/").removeprefix("http://dx.doi.org/").strip()
        ),
    )
    chave_dup("title", lambda v: re.sub(r"\W+", " ", v.lower()).strip())

    nao_usadas = sorted(set(bib) - citadas - registradas)
    if nao_usadas:
        avisos.append(
            "entradas no .bib que não são citadas por nenhum dos .tex: " + ", ".join(nao_usadas)
        )

    print()
    print(f"arquivos .tex conferidos : {', '.join(p.name for p in tex_paths)}")
    print(f"entradas no .bib         : {len(bib)}")
    print(f"chaves citadas           : {len(citadas)} (+{len(registradas)} com \\nocite)")
    print(f"erros                    : {erros}")
    print(f"avisos                   : {len(avisos)}")
    for a in avisos:
        print(f"  AVISO: {a}")
    if erros:
        print("\nRESULTADO: há erros de citação/referência (veja acima).")
        return 1
    print(
        "\nRESULTADO: toda citação tem entrada no .bib e toda entrada tem os campos obrigatórios."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
