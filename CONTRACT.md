# Reproduction Contract

## Paper
- **Title**: MCFST: Spatial domain identification method based on multi-view graph convolutional network and graph fusion network
- **Authors**: Zilong Zhang, Hao Duan, Xin Gao
- **Venue**: Bioinformatics, 2026
- **Official Code**: https://github.com/dw666666/MCFST

## Selected Experiment
**Human breast cancer spatial domain identification (Section 2.3)**

MCFST identifies spatial domains (20 fine-grained subregions) on a human breast cancer 10X Visium dataset and compares against SEDR, BayesSpace, SpaGCN, SpaNCMG, GraphST, and MNMST.

## Rationale
- Data and code are both available in the official repo
- The dataset is self-contained (3798 spots, 20 annotated subregions)
- The task is well-defined: clustering spots into spatial domains, evaluated via ARI against manual annotations
- The paper reports a single concrete metric (ARI) that is easy to verify
- This is the first real-data experiment in the paper and represents the core claim

## Paper-Reported Metric
- **ARI = 0.693** for MCFST on the human breast cancer dataset
- Baselines: MNMST (0.58), all others (<0.55)

## Data Provenance
- The official repo includes pre-packaged data under `data/`:
  - `v1_filtered_feature_bc_matrix.h5` — gene expression matrix (10X Visium)
  - `metadata.tsv` — spot annotations with `fine_annot_type` (20 subregions) as ground truth
  - `spatial/` — tissue images, spot positions, scale factors
- The code's `dataProcess.py` preprocesses this into `preprocessed_datav1/input/` with:
  - Graph adjacency matrices (A1, A2, AC)
  - Mutual information matrices
  - PCA-reduced features
  - Label array

## Required Artifacts
1. Preprocessed data under `preprocessed_datav1/input/`
2. Trained MCFST model
3. Predicted cluster labels (`.npy`)
4. ARI metric computed from predictions vs. ground truth
5. Run log capturing all 10 seeds
6. Verification script that recomputes ARI from saved predictions
7. At least 2 simple baselines (K-means on PCA features, Louvain clustering)
8. Plots: UMAP/t-SNE of learned embeddings, spatial domain map

## Acceptance Criteria
1. Pipeline runs end-to-end without manual intervention
2. ARI is computed correctly using `sklearn.metrics.adjusted_rand_score`
3. All 10 seeds are run and reported (mean ± std, median, min, max, best)
4. Verification script matches reported ARI values
5. Baselines are run on the same data
6. No data fabrication — all metrics are recomputed from saved outputs

## Known Issues / Blockers
- `dataset.py` is imported in `main_v4.py` but not present in the repo; may need to be created or the import removed
- Package version differences (torch 2.1 vs 1.13, networkx 3.x vs 2.x, etc.) may require compatibility patches
- dgl 2.2.0 API may differ from what the original code expects
- Git operations are blocked by sandbox; commits will be noted but may not be possible
