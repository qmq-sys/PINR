# PINR Experiment Report

## Dataset

- subject number (train+val+test listed): train=3, test=1
- sampling ratio: 1.0
- training subjects: 101309, 102715, 103515
- testing subjects: 106319
- experiment: `E:\BaiduNetdiskDownload\Population-DTI-INR\experiments\population_dti\20260829_174239`
- checkpoint: `E:\BaiduNetdiskDownload\Population-DTI-INR\experiments\population_dti\20260829_174239\checkpoints\best\theta.pt`

## Signal Reconstruction

- average PSNR: 6.4300
- average SSIM: 0.9729
- average Relative MSE: nan

**Interpretation:** Signal reconstruction quality measures how well predicted DWI intensities match observations (S0-normalized when configured). Higher PSNR/SSIM and lower RelMSE indicate better signal fit.

## Microstructure Reconstruction

- FA MAE: 0.266846
- FA Correlation: -0.0418
- MD MAE: 4.385289e-04
- MD Correlation: -0.0078

**Interpretation:** Whether PINR recovers diffusion tensor information is judged primarily by FA/MD agreement with WLS reference on brain ∩ WLS_valid. High Pearson correlation with low MAE suggests microstructure recovery.

## Population Generalization

### Zero-shot (z = 0)

- n subjects: 1
- mean PSNR: 6.3182
- mean SSIM: 0.9739
- mean Relative Error (RelMSE): nan
- mean FA_MAE: 0.266147
- mean FA_CC: -0.0412
- mean MD_MAE: 4.366102e-04
- mean MD_CC: -0.0033

### Latent adaptation

- n subjects: 1
- mean PSNR: 6.4300
- mean SSIM: 0.9729
- mean Relative Error (RelMSE): nan
- mean FA_MAE: 0.266846
- mean FA_CC: -0.0418
- mean MD_MAE: 4.385289e-04
- mean MD_CC: -0.0078


### Comparison

| Mode | FA_CC | MD_CC | PSNR |
|------|------:|------:|-----:|
| Zero-shot | -0.0412 | -0.0033 | 6.3182 |
| Adaptation | -0.0418 | -0.0078 | 6.4300 |

**Interpretation:** Whether population latent improves unseen subject prediction is shown by adaptation vs zero-shot FA/MD correlation and signal metrics. Improved signal with degraded FA still indicates an objective mismatch.

## Conclusion

Further optimization required.

- Decision threshold: FA correlation > 0.85 → success (observed FA_CC=-0.0418).
