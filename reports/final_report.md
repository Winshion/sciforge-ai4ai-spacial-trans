# Final Report

## Paper-reported metric
MCFST ARI = 0.693 on Section 2.3 / Fig. 3A human breast cancer, 20 annotated subregions.

## Best reproduced run
Seed 2, epochs 20, ARI 0.386956.

## Full-run mean +/- std
Seeds: [0, 1, 2]
All ARIs: [0.370811, 0.314713, 0.386956]
Mean +/- std: 0.357493 +/- 0.037918
Median: 0.370811
Min/max: 0.314713 / 0.386956

## Selected subset result
No subset selected; all completed fixed-seed runs are reported.

## Baseline results
- KMeans seed 0: ARI 0.509272, predicted clusters 20.
- KMeans seed 1: ARI 0.522217, predicted clusters 20.
- KMeans seed 2: ARI 0.500898, predicted clusters 20.
- Louvain seed 0: ARI 0.433754, predicted clusters 17.

## Instability analysis
MCFST run variance is reported above; the verdict is based on the full distribution, not the best seed alone.

## Blockers
- Original `.git_repro` and contract files disappeared during execution; no destructive reset was performed. A new `.git_repro` was initialized only if needed for final checkpointing.
- Previous local preprocessed arrays were unavailable after that workspace change; this reproduction rebuilt features/graphs from official raw H5 and metadata, so it is not bit-identical to the paper preprocessing.
- Official `main_v4.py` is CUDA-assumptive; CPU wrapper used official model architecture with rebuilt graph views.

## Reviewer verdict
Audit pass: True; failures: [].

## Latest Git commit
02a6e6e4cb439e8fdafa56480cae63e135a34902

## Final verdict
partially reproduced
