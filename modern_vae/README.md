# Multimodal recurrent VAE (synthetic data)

A standalone, CPU-executable PyTorch implementation inspired by the multimodal coronary angiography anomaly-detection publication linked in the root README. This is a new demonstration, not the exact published architecture/configuration, a clinical device, or a reproduction of the reported clinical results. It does not alter `VAE_reproducible_improved.py` or the historical scripts.

## Architecture and objective

Two aligned, variable-length feature sequences (RGB and pose) are encoded with separate LSTMs. The concatenated final valid hidden states parameterize the mean and diagonal log-variance of a shared Gaussian latent variable. The model samples that variable for training and uses the posterior mean for deterministic validation and scoring. Two distinct LSTM decoders reconstruct the modality-specific sequences. The inputs are standardized and the reconstruction heads are linear.

The encoders use packed sequences so padded suffixes do not enter the latent state. The objective consists of the mean of the two per-case masked reconstruction MSEs plus beta times the mean KL divergence from the standard normal prior. Averaging over valid time and feature dimensions for each modality gives both streams equal weight regardless of their dimensions. The objective and scaling differ from the historical script. The anomaly score is the average masked reconstruction MSE over the two modalities, not the KL term.

This is whole-procedure retrospective detection, not an online or causal anomaly detector. It can operate on aligned feature tensors with configurable dimensions, but the demonstration does not extract real video/pose features or load clinical data.

## Data and leakage prevention

Artificial feature vectors are generated using a shared low-dimensional temporal signal, independent linear projections for each modality, and noise. Anomalies are produced by deliberately large artificial perturbations; they do not simulate catheter exchanges. The dimensions and durations are intentionally much smaller than the published clinical data so training is inexpensive.

The seeded split reserves 50% of normal cases for training, approximately 10% for validation, and divides the remaining normals between calibration and test. Anomalies are independently split between calibration and test. All splits contain distinct procedure IDs. Standardization uses only valid frames from normal training cases, before padding. The validation set selects the training epoch; the labeled calibration set selects the threshold that maximizes F1, breaking ties with the highest threshold. A case is classified as anomalous if its score is strictly greater than that threshold. The held-out test set is evaluated once, without fitting any model, scaler or threshold on it.

Unsupervised here refers to representation learning on normal cases; the operating threshold is selected using labeled calibration examples. The result reports F1, precision, recall, ROC AUC and average precision; average precision is not labeled as trapezoidal PR AUC.

## Run from the repository root

Use a separate Python 3.11 or 3.12 environment. CI uses Python 3.11; the original feature-extraction dependencies and submodule are not needed.

```bash
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r modern_vae/requirements.txt
python -m pytest modern_vae/tests -q
python -m modern_vae --normal 64 --anomalies 32 --epochs 12 --seed 42 --output /tmp/vae_synthetic.json
```

The default uses 12 RGB features, 6 pose features, 32 hidden units, latent dimension 8 and beta 0.01. `python -m modern_vae --help` lists options. A shorter smoke run is `python -m modern_vae --normal 20 --anomalies 8 --epochs 2 --hidden 8 --output /tmp/vae_smoke.json`. The runner uses CPU, one thread, fixed initialization and deterministic PyTorch algorithms. Results can vary slightly across library versions. It records the split IDs, train-only scaler fit IDs, loss history, best epoch, threshold and held-out synthetic scores; weights are not saved.

## Example results

The checked-in [JSON report](results/synthetic_example.json) uses the default configuration and PyTorch 2.10.0+cpu. There are 13 normal and 16 anomalous held-out synthetic cases.

| Metric | Held-out synthetic data |
| --- | ---: |
| F1 | 0.970 |
| Precision | 0.941 |
| Recall | 1.000 |
| ROC AUC | 1.000 |
| Average precision | 1.000 |

The synthetic anomalies are intentionally obvious to exercise the complete pipeline. These values cannot be compared with the clinical study's results, are not evidence of real-world generalization, and should not be interpreted as a clinical performance claim. Tests cover disjoint splits, normal-only scaling, invariance to padding, gradients through both encoders and decoders, masked scores, calibration isolation, and repeatability of a short training run. A dedicated GitHub Actions workflow runs the new module and tests without hospital data or legacy extraction dependencies.

## Next steps

Recover the original authorized two-stream network, data/split definitions and exact experimental settings; add real-data adapters only when access is permitted. Evaluate subtler synthetic perturbations and ablations before any larger experimental claims. Do not distribute hospital features, patient-level annotations, private paths or checkpoints without authorization. The current demonstration has no loading of untrusted pickle data.
