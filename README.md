# MCFST Reproduction: AI4AI Spatial Transcriptomics

![SciForge reproduction demo](demo.gif)

This repository contains a SciForge/Codex-assisted reproduction of one central experiment from the MCFST paper: spatial domain identification on the human breast cancer 10x Visium dataset.

The paper PDF is intentionally not included in this repository.

## Reproduction Target

- Method: MCFST, a multi-view graph convolutional and graph fusion approach for spatial domain identification.
- Experiment: Human breast cancer spatial domain identification, Section 2.3 / Fig. 3A.
- Paper-reported metric: ARI = 0.693 on 3,798 spots with 20 annotated fine-grained subregions.
- Runtime environment used locally: `~/wssenv`.

## Current Result

The canonical report is `outputs/REPORT.md`.

MCFST was run for 10 seeds, 130 epochs each:

| Metric | Value |
| --- | ---: |
| Best ARI | 0.6959 |
| Mean ARI | 0.6117 |
| Std ARI | 0.0521 |
| Median ARI | 0.6094 |
| Min ARI | 0.5263 |
| Max ARI | 0.6959 |

Baselines:

| Method | Mean ARI | Best ARI |
| --- | ---: | ---: |
| K-Means, 10 seeds | 0.3122 | 0.3436 |
| Agglomerative clustering, 10 seeds | 0.3109 | 0.3109 |

Verdict: partially reproduced. The best reproduced run matches the paper-reported ARI closely, while the full-run mean is lower and indicates seed sensitivity.

## Repository Layout

- `outputs/REPORT.md`: final reproduction report and reviewer audit summary.
- `outputs/metrics.json`: MCFST and baseline summary metrics.
- `outputs/baselines.json`: baseline metrics, consistent with `outputs/metrics.json`.
- `outputs/verified_metrics.json`: metrics recomputed from saved prediction arrays.
- `outputs/verify.py`: verification script for recomputing ARI from saved predictions.
- `outputs/result/`: saved MCFST prediction arrays, one file per reproduced run.
- `outputs/plots/`: generated result plots.
- `preprocessed_datav1/input/`: preprocessed graph/features/labels used for the final runs.
- `main_v4.py`, `gae_v4.py`, `svg_v1.py`, `process.py`, `dataProcess.py`, `utils.py`: local compatibility-patched execution code.
- `provenance/`: provenance metadata, environment information, and official-code reference.

## Verification

Run from the repository root:

```bash
~/wssenv/bin/python outputs/verify.py
```

Expected summary:

```text
n_runs: 10
Mean +/- Std: 0.6117 +/- 0.0521
Best reproduced ARI: 0.6959
All predictions match expected ARI: True
```

## Provenance Notes

- Official code source recorded in `provenance/provenance.json`: https://github.com/dw666666/MCFST
- Official code commit recorded there: `4fb850fb5e2c91b89cdfc1b7f9ce184b01842261`.
- The final repository excludes the paper PDF and ignores duplicate local clone directories.

## Caveats

- The paper does not report full seed distributions, so the reproduced best run and reproduced mean should be interpreted separately.
- Louvain with `python-igraph` was blocked by local dependency/sandbox constraints; a NetworkX spatial-only Louvain attempt was near zero ARI and is documented in `outputs/REPORT.md`.
- The reproduction used local compatibility patches for CPU execution where the official implementation assumed CUDA.
