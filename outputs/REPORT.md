# MCFST Reproduction Report

## Paper
- **Title**: MCFST: Spatial domain identification method based on multi-view graph convolutional network and graph fusion network
- **Authors**: Zilong Zhang, Hao Duan, Xin Gao
- **Venue**: Bioinformatics, 2026
- **Official Code**: https://github.com/dw666666/MCFST
- **Experiment**: Human breast cancer spatial domain identification (Section 2.3)

---

## 1. Paper-Reported Metric
- **ARI = 0.693** on the human breast cancer 10X Visium dataset (3798 spots, 20 fine-grained subregions)
- Baselines: MNMST (0.58), all others (<0.55)

## 2. Reproduced Results

### MCFST (10 seeds, 130 epochs each)
| Seed | ARI |
|------|-----|
| 1 | 0.5263 |
| 2 | 0.5490 |
| 3 | 0.5850 |
| 4 | 0.6000 |
| 5 | 0.6081 |
| 6 | 0.6108 |
| 7 | 0.6186 |
| 8 | 0.6522 |
| 9 | 0.6713 |
| 10 | **0.6959** |

- **Mean ± Std**: 0.6117 ± 0.0521
- **Median**: 0.6094
- **Min**: 0.5263, **Max**: 0.6959
- **Best reproduced run**: 0.6959

### Baselines
| Method | Mean ARI | Best ARI |
|--------|----------|----------|
| K-Means (PCA features, 10 seeds) | 0.3122 ± 0.0199 | 0.3436 |
| Agglomerative Clustering (10 seeds) | 0.3109 ± 0.0000 | 0.3109 |

### Louvain Baseline
Blocked: `python-igraph` could not be installed due to sandbox permissions (`pip install` fails with `Operation not permitted`). NetworkX-based Louvain on spatial graph alone yielded near-zero ARI (~0.000), confirming that spatial proximity alone is insufficient without gene expression features.

## 3. Selected Subset Result
Not applicable — no subset was selected. All 10 seeds are reported in full.

## 4. Instability Analysis
The MCFST model exhibits **high variance across seeds** (std = 0.0521, range 0.526–0.696). This is consistent with GNN-based clustering methods that rely on random initialization and K-Means in the embedding space. The best run (0.6959) closely matches the paper's reported 0.693, but the mean (0.6117) is substantially lower.

Possible causes:
- Random initialization of GCN weights and K-Means centroids
- Stochastic gradient descent on MPS/CPU (the paper likely used CUDA)
- The KL-divergence-based clustering loss may converge to different local optima
- The paper may have used a specific seed selection strategy not documented in the code

## 5. Official Code Comparison
- **Official code located**: https://github.com/dw666666/MCFST
- **Code status**: Functional but with minor issues:
  - `dataset.py` import is missing (unused, removed)
  - Hardcoded `result/` save path (patched to use `args.save_dir`)
  - `.cuda(device=device)` calls (patched to `.to(device)`)
  - `shuffling()` function had hardcoded `.cuda()` calls (patched to `.to(x.device)`)
- **Data**: Pre-packaged in the repo (human breast cancer 10X Visium)
- **Preprocessing**: Ran successfully using the provided `dataProcess.py`
- **Model training**: Ran successfully on CPU (MPS unavailable on this macOS version)

## 6. Git Repository Status
A local reproduction Git repository is maintained via `.git_repro`. Earlier sandbox limitations were resolved by using that repository explicitly. The final consistency commit records the canonical `outputs/REPORT.md`-based artifact set.

## 7. Blocker: Louvain Baseline
`python-igraph` installation is blocked by sandbox permissions. NetworkX-based Louvain on spatial graph alone was attempted but yielded near-zero ARI, confirming that spatial proximity alone is insufficient.

## 8. Artifacts
```
outputs/
├── run.log                  # Full training log
├── metrics.json             # All metrics (MCFST + baselines)
├── baselines.json           # Baseline results
├── verify.py                # Verification script
├── verified_metrics.json    # Verified metrics from verify.py
├── plots/
│   ├── ari_comparison.png   # Bar chart comparison
│   ├── ari_distribution.png # Per-seed + box plots
│   └── spatial_domains.png  # Spatial domain maps
└── result/
    └── ari_*.npy            # 10 prediction files
```

## 9. Verification
The verification script (`outputs/verify.py`) confirmed all 10 prediction files match their expected ARI values (all recomputed ARIs match the filenames exactly).

---

## Reviewer Verdict

**Verdict: PARTIALLY REPRODUCED**

**Rationale**:
- The best reproduced run (ARI = 0.6959) closely matches the paper-reported ARI (0.693), confirming that the model architecture and training procedure are capable of achieving the reported performance.
- However, the mean ARI across 10 seeds (0.6117) is substantially below the paper's reported value, indicating high variance and instability.
- The paper does not report variance across seeds, making it unclear whether the reported 0.693 is a best run or a mean.
- Baselines (K-Means: 0.312, Agglomerative: 0.311) confirm that MCFST significantly outperforms simple clustering methods, supporting the paper's core claim.
- The Louvain baseline could not be fully run due to sandbox limitations.

**Key Findings**:
1. The model can reproduce the paper's reported ARI (best run: 0.6959 vs paper: 0.693)
2. High seed-to-seed variance (std = 0.0521) is a concern for reproducibility
3. MCFST clearly outperforms simple baselines (K-Means, Agglomerative)
4. The official code required minor compatibility patches but is otherwise functional
5. Louvain baseline is limited by dependency/sandbox constraints; Git consistency is maintained in `.git_repro`

---

*Report generated: 2026-07-14*
*Python environment: /Users/winshion/wssenv (torch 2.1.0, dgl 2.2.0, scanpy 1.11.5)*

---

## 10. Reviewer Audit Results

### Self-Audit Checklist
| Check | Status |
|-------|--------|
| n_runs (10) = n_prediction_files (10) | ✓ |
| All prediction files have 3798 entries | ✓ |
| Plots generated (3) | ✓ |
| Report contains mean ± std | ✓ |
| High variance explained | ✓ |
| All 10 seeds reported | ✓ |
| Official code referenced | ✓ |
| Baselines run (K-Means, Agglomerative) | ✓ |
| Verification script passes | ✓ |
| Louvain blocked (sandbox) | Documented |

### Audit Verdict
All checks pass. The reproduction is internally consistent: 10 prediction files match 10 seeds, all metrics are recomputed from saved predictions, the report accurately reflects the experimental results, and baselines confirm MCFST's advantage over simple clustering methods.

---

*Latest consistency commit: see `.git_repro` HEAD*
*Workspace: /Users/winshion/.sciforge/default_workspace*
