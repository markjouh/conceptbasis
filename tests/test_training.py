import sys

import pytest
import torch

from conceptbasis.train import adapted, evaluate, parse_args


@pytest.mark.parametrize(
    ("objective", "expected_lambda"),
    [("contrastive", 0.0), ("group-mean", 8.0), ("reverse-ridge", 512.0)],
)
def test_objective_defaults_are_explicit(monkeypatch, objective, expected_lambda):
    monkeypatch.setattr(sys, "argv", ["train", "--objective", objective])

    args = parse_args()

    assert args.lambda_orth == expected_lambda
    assert objective.replace("-", "_") in args.run_name


def test_adapted_restores_adapter_mode():
    adapter = torch.nn.Identity()
    features = torch.randn(4, 3)
    adapter.eval()

    adapted(adapter, features, "cpu")

    assert not adapter.training


def test_evaluation_retrieval_uses_full_input():
    adapter = torch.nn.Identity()
    image = torch.eye(3)
    text = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    scores = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = evaluate(adapter, adapter, image, text, scores, "cpu")

    assert result["R@k"][1] == pytest.approx(2 / 3)
