# Reproducibility and code audit

This document compares the **public repository at the time of this documentation update** against Frassini *et al.*, *Minimally Invasive Therapy & Allied Technologies* (2026), DOI [10.1080/13645706.2026.2705585](https://doi.org/10.1080/13645706.2026.2705585).

## What the paper reports

- **Data:** 191 coronary angiography procedures; 134 normal and 57 with an additional catheter exchange. The recordings/poses are not distributed in this repository.
- **Temporal segmentation:** 10-second clips.
- **Feature extraction:** 512-dimensional pose features from PoseConv3D and 2048-dimensional RGB video features from a 3D CNN backbone.
- **Paper's temporal model:** separate modality-specific LSTM encoders, fusion of their final hidden states, Gaussian latent representation, and LSTM decoder.
- **Training:** reconstruct normal procedures with a VAE reconstruction objective plus beta-weighted KL divergence.
- **Thresholding:** choose a reconstruction-error cutoff by maximizing F1 on a separate **labeled calibration set**; assess performance on a separate test set.
- **Reported multimodal performance:** F1 0.83, AUC-ROC 0.88, AUC-PR 0.90.

These are **published results**, not values recomputed by the repository's CI or synthetic smoke test.

## Main code entry points

The research scripts are rooted in a single-workstation project and may need adaptation before use:

1. `videos_chunks.py` uses local `ffmpeg.exe` and `ffprobe.exe` paths to segment `.mkv` videos.
2. `convert_all_json_to_pyskl_10sclip.py` converts pose-estimation JSON into PySKL clip annotations.
3. `create_conf_files.py` and `create_conf_files_videos.py` generate PySKL config files, using absolute paths.
4. `test.py` is a feature-extraction/inference script built on older MMCV/PySKL APIs; its name does **not** mean it is an automated test.
5. `concatenate_pose_rgb.py` concatenates feature vectors from two streams for each clip.
6. `VAE_reproducible_improved.py` loads serialized feature sequences and phase annotations; defines the single-stream VAE, train/validation/calibration/test splits, threshold selection, and scoring/plots.

The final research training script expects feature files (`.pkl`) and phase-annotation CSV files with matching base names, using a `Phase` column containing `A` and `M` markers. It pads or truncates to `Config.seq_len` (default 574) and assumes a per-clip feature dimension of `Config.feature_dim` (default 2048). Use only verified, authorized inputs.

## Installing and running the lightweight smoke test

Run from the repository root, without the PySKL feature-extraction stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vae.txt
python scripts/smoke_vae.py
```

The smoke test checks shape consistency, a finite forward/loss/backward pass and elementary synthetic threshold calibration. It **does not** load clinical files, perform a full training experiment, or certify the published metrics. It may take some time to install CPU/GPU PyTorch wheels, depending on your platform.

## Known implementation caveats

These findings are based on reading the committed code; they are **not clinical-performance re-evaluation**.

| Priority | Issue | Why it matters / suggested fix |
|:--|:--|:--|
| **High** | **Scaler fitted before split** in `get_normal_data_loaders`: a `MinMaxScaler` is fitted to **all** normal procedures, including calibration/test, before random splitting. | This leaks distributional information across splits. For independent future experiments, first determine procedure-level splits, fit the scaler on *training normal procedures only*, then transform validation/calibration/test. This will change learned parameters and reported metrics; preserve historical results separately. |
| **High** | **Paper-model mismatch:** the paper specifies pose- and video-specific LSTM encoders with intermediate fusion, whereas `VAE_reproducible_improved.py` defines a single `encoder_lstm`. | Do not describe the present script as a fully reproduced implementation of the published multi-encoder architecture. Publish the appropriate two-stream model and checkpoint only if authorized and validated. |
| **High** | **Data and weights unavailable:** hospital data, tracking outputs and trained models are not public. | Distinguish publication results from repository tests. Use only de-identified, legally shareable samples or explicitly synthetic fixtures. Do not create synthetic data and claim it reproduces clinical performance. |
| Medium | **Hard-coded Windows paths** throughout research scripts and no declarative run manifest. | Read file locations from command-line arguments or environment variables. The optional `scripts/patch_legacy_vae.py` modifies the *single-stream VAE script only*, with a backup. |
| Medium | **Legacy extractor dependencies** (`mmcv-full==1.5.0`, `mmdet==2.23.0`, `mmpose==0.24.0`) are from older frameworks. | Preserve the original `requirements.txt` for provenance; install a mutually compatible historical stack for PySKL if you have the corresponding data/checkpoints, preferably in an isolated environment. |
| Medium | **Padding typo** in the `repeat_last` branch of the dataset loader: repeats `feature_dim - T_new` rather than `seq_len - T_new`. | The optional patch script applies this narrow correction. The default training configuration uses `padding='zero'`; any changed behavior must be validated. |
| Medium | **Scheduler compatibility:** `ReduceLROnPlateau(verbose=True)` is unsupported in recent PyTorch versions. | Optional patch removes that argument; the learning-rate scheduler itself is retained. |
| Medium | **Default run state** enables reproduce-specific-run mode with session timestamp `0`. | Without previously saved checkpoints at that timestamp, executing the script cannot reproduce a model. Optional patch allows mode and session selection through environment variables, but pretrained weights and feature data remain necessary. |
| Medium | **Unsafe deserialization:** scripts read `.pkl` data and scaler objects using `pickle.load`. | Treat pickle files as executable content and only load them from trusted, authorized sources. |
| Low | `process_videos_optimized_2.py` contains multiple historical processing sections, and `test.py` is not a unit test. | Separate reusable preprocessing utilities from orchestration and rename tests when refactoring; preserve historical behavior until validated. |

### Methodology reminder: what is “unsupervised” here?

The VAE learns a reconstruction representation using the normal set **without training on anomalous labels**. A labeled calibration set is then used to maximize F1 when selecting the decision threshold. Therefore the entire operational detection pipeline is **not fully label-free**; future comparisons should state precisely which stages use labels.

### Recommended steps toward research reproducibility

1. Obtain permission to access recordings, annotations, pretrained pose/RGB backbones and the model checkpoints, and establish publication-safe sharing rights for any derived features.
2. Recover the **actual two-stream model** and configurations corresponding to the paper results; separate pose- and RGB-only ablations.
3. Record stable procedure-level split manifests, preprocessing statistics computed on training data only, model/config hashes, seed, framework versions and hardware.
4. Implement deterministic padding masks when necessary so padded frames do not distort reconstruction loss; audit threshold optimization separately from test metrics.
5. Compare independent test metrics against published values **only after** the exact experimental configuration and permitted dataset have been recovered.

## Optional compatibility patch for the existing single-stream script

Run from the repository root:

```bash
python scripts/patch_legacy_vae.py --check
python scripts/patch_legacy_vae.py --apply
```

The patch saves a `.bak` copy, makes the VAE paths configurable with environment variables (`CAG_NORMAL_FEATURES`, `CAG_NORMAL_ANNOTATIONS`, `CAG_ANOMALY_FEATURES`, `CAG_ANOMALY_ANNOTATIONS`, `CAG_RUNS_DIR`), allows `CAG_MODE=search|reproduce` and `CAG_SESSION` for historical entry points, fixes the `repeat_last` padding typo, and removes the unsupported scheduler `verbose` parameter. **It does not address the scaler leakage or implement the paper's two-stream architecture.** Apply only after reviewing `git diff`. It is not required for the smoke test.

Do **not** publish clinical file paths, real raw/derived datasets or private checkpoints. `git status` should be inspected carefully before every push.

## Licensing and attribution

The **published paper** is CC BY 4.0. This is a publication license, **not a blanket software license**. The `pyskl/` directory is an external git submodule with its own license; `test.py` contains OpenMMLab notices. If this repository is to be openly licensed, determine which files are original and which are adapted from external work before choosing and adding a repository software license.

## Reference

Frassini, E., Butler, R. M., Hendriks, B. H. W., van der Elst, M., & van den Dobbelsteen, J. J. (2026). *Unsupervised multi-modal variational autoencoder for anomaly detection in coronary angiography*. **Minimally Invasive Therapy & Allied Technologies**. https://doi.org/10.1080/13645706.2026.2705585
