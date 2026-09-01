# Experiment aggregate (mean ± std)

- N subjects (ok): **3**
- Parameter MAE/RMSE/r = agreement vs WLS reference (not GT error)

| Metric | mean ± std | median |
|--------|-----------:|-------:|
| FA_MAE | 0.178363 ± 0.00826778 | 0.178902 |
| FA_RMSE | 0.249011 ± 0.0103995 | 0.250043 |
| FA_r | 0.247245 ± 0.0406828 | 0.228993 |
| MD_MAE | 0.00022084 ± 3.00493e-05 | 0.000228259 |
| MD_RMSE | 0.000396136 ± 5.32561e-05 | 0.00040686 |
| MD_r | 0.574045 ± 0.0451255 | 0.588556 |
| AD_MAE | 0.000316548 ± 2.80271e-05 | 0.000307143 |
| AD_RMSE | 0.000535403 ± 5.49229e-05 | 0.000559324 |
| AD_r | 0.451558 ± 0.141854 | 0.46721 |
| RD_MAE | 0.000252743 ± 2.98022e-05 | 0.000265642 |
| RD_RMSE | 0.000395944 ± 5.74237e-05 | 0.000414965 |
| RD_r | 0.584814 ± 0.0277282 | 0.5946 |
| DWI_MAE | 436.179 ± 16.2754 | 440.052 |
| DWI_RelMSE | 0.0881251 ± 0.0138855 | 0.0899454 |
| final_loss | 0.157244 ± 0 | 0.157244 |
| best_loss | 0.116761 ± 0 | 0.116761 |
| training_time_sec | 468.552 ± 0 | 468.552 |

## Shared INR MVP notes

- One shared network + `3` subject embeddings (latent table)
- Training: all subjects update the same θ and their z_s each epoch
- Evaluation: same `brain & WLS_valid`, seed=42, max_voxels=131072 as Independent INR

### Question 1 — training stability
- Check `best.pt`, `summary.csv`, and per-epoch logs for NaN / divergence.

### Question 2 — DWI reconstruction vs Independent
- Shared DWI RelMSE: mean=0.0881251, median=0.0899454, std=0.0113374
- See `comparison_independent_vs_shared.csv` for per-subject Δ (Shared − Independent).

### Question 3 — subject-to-subject variability
- Shared FA MAE spread: mean=0.178363, median=0.178902, std=0.00675061

### Question 4 — prior Independent failure subjects

| subject | Shared DWI RelMSE | Shared FA MAE | note |
|---------|------------------:|--------------:|------|

### Question 5 — new failures
- Subjects with highest Shared FA MAE or DWI RelMSE vs cohort median should be reviewed manually.

### Focus subjects (not hard-coded failures)

- **101309**: DWI_RelMSE=0.0734192, FA_MAE=0.1863, MD_MAE=0.000187776
