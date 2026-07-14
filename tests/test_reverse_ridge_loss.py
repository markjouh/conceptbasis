"""Tests for the production reverse-ridge estimator."""

import numpy as np
import torch

from conceptbasis.losses import reverse_ridge_objective


def test_reverse_ridge_recovers_partial_effects_with_correlated_attributes():
    rng = np.random.default_rng(11)
    first = rng.integers(0, 2, size=4096)
    agree = rng.random(4096) < 0.8
    second = np.where(agree, first, 1 - first)
    labels = torch.tensor(np.stack([first, second], axis=1), dtype=torch.float32)
    true_effects = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    embeddings = labels @ true_effects + torch.randn(4096, 2) * 0.01

    result = reverse_ridge_objective(
        embeddings,
        labels,
        torch.ones_like(labels),
        alpha=1e-4,
    )

    cosine = float(result["directions"][0] @ result["directions"][1])
    assert abs(cosine) < 0.02
    assert float(result["explained_fraction"]) > 0.99


def test_reverse_ridge_is_differentiable_with_uncertainty():
    torch.manual_seed(5)
    embeddings = torch.randn(128, 8, requires_grad=True)
    labels = torch.randint(0, 2, (128, 4), dtype=torch.float32)
    observed = torch.ones_like(labels)
    observed[:12, 2] = 0

    result = reverse_ridge_objective(embeddings, labels, observed, alpha=1e-3)
    (result["reconstruction"] + result["orth"]).backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
