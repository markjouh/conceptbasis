import unittest

import torch
import torch.nn.functional as F

from conceptbasis.losses import (
    GroupMeanOrthogonalityLoss,
    ReverseRidgeOrthogonalityLoss,
    symmetric_clip_loss,
)
from conceptbasis.models import Adapter


class LossTests(unittest.TestCase):
    def test_symmetric_clip_loss_matches_definition(self):
        image = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
        text = image.clone()
        scale = torch.tensor(2.0)
        logits = scale * image @ text.T
        target = torch.arange(2)
        expected = 0.5 * (
            F.cross_entropy(logits, target)
            + F.cross_entropy(logits.T, target)
        )
        self.assertTrue(torch.allclose(symmetric_clip_loss(image, text, scale), expected))

    def test_weighted_orthogonality_is_finite(self):
        loss = GroupMeanOrthogonalityLoss(n_concepts=3, dim=4)
        weights = torch.ones(3, 3) - torch.eye(3)
        loss.set_pair_weights(weights)
        embedding = torch.randn(12, 4, requires_grad=True)
        labels = torch.rand(12, 3)
        value = loss(embedding, labels)["orth"]
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertIsNotNone(embedding.grad)

    def test_reverse_ridge_module_is_differentiable(self):
        loss = ReverseRidgeOrthogonalityLoss(alpha=1e-3)
        embedding = torch.randn(128, 8, requires_grad=True)
        labels = torch.randint(0, 2, (128, 4), dtype=torch.float32)
        result = loss(embedding, labels, torch.ones_like(labels))
        result["orth"].backward()
        self.assertIsNotNone(embedding.grad)
        self.assertTrue(torch.isfinite(embedding.grad).all())

    def test_adapter_shape(self):
        adapter = Adapter(d_in=8, d_out=5, hidden=16)
        self.assertEqual(adapter(torch.randn(3, 8)).shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
