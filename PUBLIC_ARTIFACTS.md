# Public artifacts and reproducibility policy

Public repository:

https://github.com/Crist407/Implementacio-compressor-basat-en-xarxes-neurals-per-a-us-en-satel-lit

This repository is prepared to publish source code, configuration, report
sources, model weights and the RAW crop dataset used by the public examples. It
is not intended to store the complete local experimental workspace.

## Included in Git

- `src/`: C implementation, Python reference helpers and analysis scripts.
- `scripts/`: local, Raspberry and documentation workflows.
- `config/`: thresholds, minimal lambda005 calibration and configuration tables
  used by reproducible examples.
- `weights/`: exported SORTENY weights required by the C codec.
- `data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw`: canonical RAW sample.
- `data/Sentinel2A_crop_test/`: 120 RAW crops for inspection and reproducible
  experiments.
- `data/README.md`: RAW format and visualization instructions.
- `docs/informe_final/`: final report sources, figures and bibliography.
- `requirements.txt`, `Makefile`, `README.md` and this manifest.

## Excluded from Git

- `.venv/` and local Python environments.
- `output/checkpoints/` complete experiment outputs.
- Raspberry bundles and imported `.tar.gz` archives.
- compiled C binaries and object files.
- LaTeX auxiliary files such as `.aux`, `.fls`, `.fdb_latexmk`, `.synctex.gz`.
- local institutional template copies under `INFORME FINAL/`.
- browser or Windows metadata files such as `*:Zone.Identifier`.

## Large artifacts

The complete checkpoints are intentionally not versioned because they occupy
tens of gigabytes. The report records the checkpoint names associated with each
result. When a large artifact is needed, regenerate it with the scripts in
`src/python/analysis/` and `scripts/raspberry/`, or restore it under
`output/checkpoints/` from the archived experimental storage.

The RAW dataset is versioned directly in Git because each file is approximately
4 MiB and remains below GitHub's per-file hard limit. The complete dataset adds
approximately 485 MiB to a clone, which is an accepted tradeoff here to make the
public examples and visual inspection self-contained.

The minimal measured route can be regenerated without external checkpoints:

```bash
python3 src/python/analysis/run_lambda005_measured_quality_route.py \
  --output-dir output/checkpoints/lambda005_measured_quality_route_smoke \
  --modes global_target_measured focus_bgq128_measured \
  --threads 4
```

Expected invariants for the smoke run:

- every Q-map is 1,024 bytes;
- each reconstructed RAW is 4,194,304 bytes;
- the final bitstreams are readable by `sorteny_decompressor`;
- `metrics.csv`, `metrics.json`, `run_meta.json` and hashes are generated.

## Recommended public staging set

For a clean public release, stage only the files needed to reproduce concrete
cases:

```bash
git add .gitignore PUBLIC_ARTIFACTS.md README.md Makefile requirements.txt
git add config/auto_thresholds_lambda005.tsv config/fq_calibration_lambda005.tsv
git add data/.gitkeep data/README.md
git add data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw
git add data/Sentinel2A_crop_test
git add weights/encoder weights/decoder
git add src/c src/python/reference src/python/utils
git add src/python/analysis/run_lambda005_measured_quality_route.py
git add src/python/analysis/audit_csmr_selected_cases_from_full.py
git add src/python/analysis/audit_csmr_experimental_threshold_modes.py
git add src/python/analysis/audit_csmr_preserve_roi_policy.py
git add src/python/analysis/build_raspberry_benchmark_report.py
git add src/python/analysis/build_raspberry_qmap_cost_report.py
git add src/python/analysis/compare_raspberry_optimized_vs_baseline.py
git add scripts/raspberry scripts/docs
git add docs/informe_final
```

Do not stage all local analysis increments by default. Many files under
`src/python/analysis/`, `docs/informes/`, `docs/informes muestra/`,
`docs/web/` and `output/` are development artifacts or intermediate reports.
Keep them local unless the final appendix or README explicitly references them
as part of a reproducible example.

Current examples of local-only material:

- historical drafts under `docs/informes/` except the already tracked previous
  report if it is cited;
- review notes and feedback summaries;
- example TFG PDFs under `docs/informes muestra/`;
- exploratory HTML tools under `docs/web/`;
- all generated checkpoints and Raspberry bundles under `output/`.

## Before publishing

Before tagging a public release, run:

```bash
python3 -m py_compile src/python/analysis/run_lambda005_measured_quality_route.py
make MODE=release OMP=1
make MODE=release OMP=1 test_ops
git diff --check
git status --short
```

The final report should cite the public repository URL and, once closed, the
exact commit or tag used for the submitted version.
