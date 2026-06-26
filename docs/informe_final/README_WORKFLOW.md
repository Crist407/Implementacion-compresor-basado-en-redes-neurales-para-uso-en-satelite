# Informe final SORTENY

Esta carpeta es la copia de trabajo de la plantilla institucional conservada en
`INFORME FINAL/`. La plantilla original no debe modificarse.

## Compilacion

Desde la raiz del repositorio:

```bash
scripts/docs/build_informe_final_compat.sh
```

El PDF se genera en `docs/informe_final/main.pdf`. El script compila la copia
de trabajo con la distribución LaTeX actual sin modificar la plantilla original
de `INFORME FINAL/`.

La emulación LaTeX 2020 solo se conserva como opción de diagnóstico:

```bash
LATEX_LEGACY_RELEASE=1 scripts/docs/build_informe_final_compat.sh
```

Limpieza de auxiliares:

```bash
scripts/docs/build_informe_final_compat.sh --clean
```

Limpieza incluyendo el PDF:

```bash
scripts/docs/build_informe_final_compat.sh --clean-all
```

## Metodo de trabajo

- `main.tex` orquesta el documento y contiene el orden de los capitulos.
- `frontpage.tex`, `abstract.tex` y `resum.tex` contienen los preliminares.
- Cada capitulo vive en `ChapterN/text_chN.tex` y empieza con `\chapter`.
- Las figuras y tablas deben tener `\label` y citarse desde el texto.
- La bibliografia se gestionara con un archivo `.bib` y el estilo `tesi.bst`
  cuando se cierre la pauta bibliografica.
- No se definiran capitulos nuevos hasta aprobar el indice y las pautas formales.

## Convenciones previstas

- Capitulos: `\label{ch:identificador}`.
- Secciones: `\label{sec:identificador}`.
- Figuras: `\label{fig:identificador}`.
- Tablas: `\label{tab:identificador}`.
- Ecuaciones: `\label{eq:identificador}`.
- Cada termino tecnico se define la primera vez que aparece.
