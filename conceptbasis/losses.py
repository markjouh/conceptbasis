"""Contrastive, group-mean, and reverse-ridge training objectives."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_clip_loss(img_n, txt_n, logit_scale):
    """Symmetric InfoNCE for aligned, L2-normalized image/text batches."""
    logits = logit_scale * img_n @ txt_n.t()
    target = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


class GroupMeanOrthogonalityLoss(nn.Module):
    """Conditional orthogonality over soft-labeled concept directions.

    Supervision is a soft score s_k in [0, 1] per image and concept (here:
    calibrated zero-shot scores from a frozen vision-language model -- no
    human labels). Each concept's direction is defined *from the batch* as the
    difference of soft-weighted class means,

        d_k = normalize( mean_{s_k}(z)  -  mean_{1-s_k}(z) ),

    which is differentiable in the embedding z, so the loss reshapes the
    encoder:

    - orthogonality: mean of squared cosines <d_j, d_k> over penalized pairs.
      By default all pairs are penalized; call `set_pair_mask` for a hard
      exemption graph or `set_pair_weights` for continuous pressure.

    Exponential-moving-average class means are kept as buffers so that a
    concept with no batch support (rare concepts, small batches) still has a
    stable direction for the Gram matrix instead of dropping out of the loss.
    """

    def __init__(self, n_concepts: int, dim: int, ema: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.n, self.dim, self.ema, self.eps = n_concepts, dim, ema, eps
        # EMA class means (detached): fallback directions for unsupported concepts
        self.register_buffer("mu_pos", torch.zeros(n_concepts, dim))
        self.register_buffer("mu_neg", torch.zeros(n_concepts, dim))
        self.register_buffer("inited", torch.zeros(n_concepts, dtype=torch.bool))

    def set_pair_mask(self, mask: torch.Tensor):
        """Restrict the orthogonality penalty to selected concept pairs.

        mask: boolean [n, n]; True = penalize this pair's overlap. Pass
        `|corrcoef(labels)| < threshold` to exempt naturally-correlated pairs
        (conditional orthogonality). Unset = penalize every pair.
        """
        self.register_buffer("pair_mask", mask.bool(), persistent=False)

    def set_pair_weights(self, weights: torch.Tensor):
        """Continuously weight pairwise orthogonality penalties.

        weights: nonnegative [n, n] matrix. The orthogonality loss is the
        weighted mean squared cosine over off-diagonal pairs. This is mutually
        exclusive with ``set_pair_mask``.
        """
        if hasattr(self, "pair_mask"):
            raise ValueError("pair mask and pair weights are mutually exclusive")
        if weights.shape != (self.n, self.n):
            raise ValueError(f"expected pair weights {(self.n, self.n)}, got {weights.shape}")
        if (weights < 0).any():
            raise ValueError("pair weights must be nonnegative")
        self.register_buffer("pair_weights", weights.float(), persistent=False)

    def forward(self, Z, S):
        """Z: [B, dim] raw (unnormalized) image embeddings. S: [B, n] soft labels.

        Returns {"orth": scalar, "D": [n, dim] detached unit concept
        directions for logging/eval}.
        """
        # Vectorized soft class means for every concept. The previous
        # per-concept Python loop forced hundreds of device synchronizations on
        # MPS per batch; these two matrix multiplies compute the same means.
        sp = S.sum(0)
        sn = (1.0 - S).sum(0)
        supported = (sp > 1e-3) & (sn > 1e-3)
        mp = (S.T @ Z) / sp.clamp(min=self.eps).unsqueeze(1)
        mn = ((1.0 - S).T @ Z) / sn.clamp(min=self.eps).unsqueeze(1)

        with torch.no_grad():
            was_inited = self.inited.clone()
            next_pos = torch.where(
                was_inited.unsqueeze(1),
                self.ema * self.mu_pos + (1 - self.ema) * mp.detach(),
                mp.detach(),
            )
            next_neg = torch.where(
                was_inited.unsqueeze(1),
                self.ema * self.mu_neg + (1 - self.ema) * mn.detach(),
                mn.detach(),
            )
            self.mu_pos.copy_(torch.where(supported.unsqueeze(1), next_pos, self.mu_pos))
            self.mu_neg.copy_(torch.where(supported.unsqueeze(1), next_neg, self.mu_neg))
            self.inited.logical_or_(supported)

        batch_delta = mp - mn
        batch_dirs = batch_delta / batch_delta.norm(dim=1, keepdim=True).clamp(min=self.eps)
        ema_delta = self.mu_pos - self.mu_neg
        ema_dirs = ema_delta / ema_delta.norm(dim=1, keepdim=True).clamp(min=self.eps)
        fallback_dirs = torch.where(self.inited.unsqueeze(1), ema_dirs, torch.zeros_like(ema_dirs))
        D = torch.where(supported.unsqueeze(1), batch_dirs, fallback_dirs)

        G = D @ D.t()                                            # pairwise cosines
        off = ~torch.eye(self.n, dtype=torch.bool, device=Z.device)
        if hasattr(self, "pair_mask"):
            off = off & self.pair_mask                           # conditional orth
        squared = G[off] ** 2
        if hasattr(self, "pair_weights"):
            weights = self.pair_weights[off]
            loss_orth = (weights * squared).sum() / weights.sum().clamp(min=self.eps)
        else:
            loss_orth = squared.mean()
        return {"orth": loss_orth, "D": D.detach()}


# Backward-compatible name used by historical checkpoints and imports.
ConceptOrthogonalityLoss = GroupMeanOrthogonalityLoss


def reverse_ridge_objective(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
    *,
    alpha: float,
    max_uncertain_fraction: float = 0.25,
    min_positives: int = 1,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Fit concept-to-embedding partial effects and measure orthogonality.

    For centered labels ``Y`` and centered embeddings ``Z``, this solves

        B = (Y.T @ Y / n + alpha I)^-1 (Y.T @ Z / n).

    Row ``B[k]`` is the regularized embedding effect of concept ``k`` while
    controlling for the other labels. Increasing ``alpha`` suppresses weakly
    identified label-covariance modes and approaches marginal group-mean
    directions, up to per-concept scale.
    """
    if embeddings.ndim != 2 or targets.ndim != 2 or observed.shape != targets.shape:
        raise ValueError("expected embeddings [B,D] and targets/observed [B,C]")
    if len(embeddings) != len(targets):
        raise ValueError("embedding and target batch sizes differ")
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")
    if not 0 <= max_uncertain_fraction < 1:
        raise ValueError("max_uncertain_fraction must be in [0, 1)")

    keep = observed.mean(dim=1) >= 1.0 - max_uncertain_fraction
    z = embeddings[keep]
    y = targets[keep]
    mask = observed[keep]
    if len(z) < 2:
        raise ValueError("too few rows remain after uncertainty filtering")

    definite = mask.sum(dim=0).clamp(min=1.0)
    prevalence = (y * mask).sum(dim=0) / definite
    filled = torch.where(mask.bool(), y, prevalence.unsqueeze(0))
    centered_y = filled - prevalence.unsqueeze(0)

    x = z - z.mean(dim=0, keepdim=True)
    # A detached scalar fixes the ridge scale without changing directions.
    rms = x.square().mean().sqrt().clamp(min=eps).detach()
    x = x / rms

    n = x.shape[0]
    label_covariance = (centered_y.T @ centered_y) / n
    label_embedding_covariance = (centered_y.T @ x) / n
    eye = torch.eye(label_covariance.shape[0], dtype=x.dtype, device=x.device)
    effects = torch.linalg.solve(
        label_covariance + alpha * eye,
        label_embedding_covariance,
    )

    reconstruction = centered_y @ effects
    fit_mse = (reconstruction - x).square().mean()
    ridge_penalty = alpha * effects.square().sum() / embeddings.shape[1]
    reconstruction_objective = fit_mse + ridge_penalty

    positive = (y * mask).sum(dim=0)
    negative = ((1.0 - y) * mask).sum(dim=0)
    effect_norm = effects.norm(dim=1)
    supported = (
        (positive >= min_positives)
        & (negative >= min_positives)
        & (effect_norm > eps)
    )
    directions = F.normalize(effects, dim=1, eps=eps)
    gram = directions @ directions.T
    pair_mask = supported[:, None] & supported[None, :]
    pair_mask.fill_diagonal_(False)
    if pair_mask.any():
        orth = gram[pair_mask].square().mean()
        orth_rms = orth.sqrt()
    else:
        orth = embeddings.sum() * 0.0
        orth_rms = orth.detach()

    explained_fraction = 1.0 - fit_mse / x.square().mean().clamp(min=eps)
    return {
        "reconstruction": reconstruction_objective,
        "fit_mse": fit_mse,
        "ridge_penalty": ridge_penalty,
        "explained_fraction": explained_fraction,
        "orth": orth,
        "orth_rms": orth_rms,
        "directions": directions,
        "effects": effects,
        "effect_norm": effect_norm,
        "supported": supported,
        "kept_rows": keep.sum(),
    }


class ReverseRidgeOrthogonalityLoss(nn.Module):
    """Module wrapper around the differentiable full-data reverse-ridge solve."""

    def __init__(
        self,
        alpha: float = 1e-3,
        max_uncertain_fraction: float = 0.25,
        min_positives: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.max_uncertain_fraction = max_uncertain_fraction
        self.min_positives = min_positives
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return reverse_ridge_objective(
            embeddings,
            targets,
            observed,
            alpha=self.alpha,
            max_uncertain_fraction=self.max_uncertain_fraction,
            min_positives=self.min_positives,
            eps=self.eps,
        )
