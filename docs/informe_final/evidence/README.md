# Extractes publics d'evidencia

Aquest directori conte extractes petits dels resultats que sostenen les xifres
principals de la memoria. No substitueix els checkpoints complets de
`output/checkpoints/`, que no es publiquen a Git per mida, pero permet revisar
les dades agregades citades al cos del document.

Fitxers:

- `csmr_policy_summary.csv`: bitrate i PSNR mitjans de `q204`,
  `adaptive_s8` i `focus_bgq128`.
- `key_metrics_summary.csv`: metriques globals de validacio, incloent
  `8.84%`, `60/60` i deltes respecte `adaptive_s8`.
- `q204_same_mask_summary.csv`: deltes ROI/fons contra `q204` amb mascara
  comuna.
- `global_target_canonical_curve.csv`: corba canonical Q--PSNR de
  `global_target`.
- `experimental_and_preserve_summary.csv`: resum dels modes experimentals i
  `preserve-roi`.
- `raspberry_operational_costs.csv`: temps i costos agregats de Raspberry.
- `qmap_cost_by_type.csv`: cost de generacio de Q-map per familia de comanda.

Els camps `archived_origin` indiquen el checkpoint local del qual es va extreure
la dada. Aquests checkpoints complets es poden regenerar seguint
l'Apendix A de la memoria, pero no formen part del repositori public.
