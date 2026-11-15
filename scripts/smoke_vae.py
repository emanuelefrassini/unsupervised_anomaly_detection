"""Small CPU synthetic-data check of the checked-in single-stream VAE.

This is NOT a reproduction of results from coronary angiography patients.
Run from repository root after installing requirements-vae.txt:
    python scripts/smoke_vae.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from VAE_reproducible_improved import (
    MinMaxScaler,
    VAE,
    evaluate_performance,
    find_optimal_threshold_f1,
    vae_loss_function,
)


def main() -> None:
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Small synthetic feature tensor, not RGB/pose data from patients.
    batch, steps, features, hidden, latent = 4, 6, 8, 12, 5
    sample = torch.rand(batch, steps, features)
    scaler = MinMaxScaler()
    scaler.fit(sample)
    scaled = scaler.transform(sample)
    assert scaled.shape == (batch, steps, features)
    assert torch.isfinite(scaled).all()
    assert float(scaled.min()) >= -1e-6
    assert float(scaled.max()) <= 1 + 1e-6
    assert torch.allclose(scaler.inverse_transform(scaled), sample, atol=1e-6)

    model = VAE(input_dim=features, hidden_size=hidden, latent_dim=latent)
    model.train()
    reconstructed, mu, logvar = model(scaled)
    assert reconstructed.shape == scaled.shape
    assert mu.shape == (batch, latent)
    assert logvar.shape == (batch, latent)
    assert torch.isfinite(reconstructed).all()

    loss, reconstruction_loss, kl_loss = vae_loss_function(
        reconstructed, scaled, mu, logvar, beta=1.0
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(reconstruction_loss)
    assert torch.isfinite(kl_loss)
    assert loss.detach().item() >= 0
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())
    assert all(
        torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.grad is not None
    )

    # Purely synthetic calibration and test scores.
    normal_calibration = np.array([0.05, 0.09, 0.13, 0.10])
    anomaly_calibration = np.array([0.61, 0.80, 0.92, 0.71])
    threshold, calibration = find_optimal_threshold_f1(
        normal_calibration, anomaly_calibration
    )
    assert float(normal_calibration.max()) <= threshold < float(anomaly_calibration.min())
    assert abs(calibration["f1"] - 1.0) < 1e-8

    normal_test = rng.uniform(0.01, 0.12, size=5)
    anomaly_test = rng.uniform(0.7, 1.0, size=5)
    metrics = evaluate_performance(normal_test, anomaly_test, threshold)
    assert metrics["overall_f1"] > 0.99, metrics

    print("PASS: scaler forward/inverse; VAE shapes; finite loss and gradients")
    print("PASS: synthetic threshold selected on calibration set")
    print("PASS: synthetic held-out score classification")
    print("Synthetic smoke test PASSED (not a clinical-data reproduction).")


if __name__ == "__main__":
    main()
