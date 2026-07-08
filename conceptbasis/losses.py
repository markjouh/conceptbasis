"""Losses for training an embedding with an explicit concept basis.

Total objective:

  L = L_contrastive + lambda_id * L_identify + lambda_orth * L_orthogonality

- L_contrastive keeps the space a retrieval embedding (image <-> caption).
- L_identify makes each concept linearly readable: for every concept there is
  a direction in the embedding along which that concept's score increases.
- L_orthogonality pushes those concept directions apart -- but only for
  concept pairs that are statistically independent in the data (conditional
  orthogonality). Naturally co-occurring concepts (e.g. `manmade`/`crafted`)
  are exempted: forcing perpendicularity on them is unsatisfiable and degrades
  the most-connected concepts.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_clip_loss(img_n, txt_n, logit_scale, keys):
    """Symmetric InfoNCE over a batch of L2-normalized image/text embeddings.

    `keys` assigns an identity to each row; off-diagonal pairs with equal keys
    are excluded from the negatives. With one unique caption per image this is
    a no-op guard; with duplicated or templated captions it prevents true
    matches from being treated as negatives.

    img_n, txt_n: [B, d] (unit norm)   logit_scale: scalar   keys: [B]
    """
    logits = logit_scale * img_n @ txt_n.t()                       # [B, B]
    b = logits.size(0)
    same = keys.view(-1, 1) == keys.view(1, -1)
    eye = torch.eye(b, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))        # false negatives
    target = torch.arange(b, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


class ConceptLossSoft(nn.Module):
    """Identification + conditional orthogonality over soft-labeled concepts.

    Supervision is a soft score s_k in [0, 1] per image and concept (here:
    calibrated zero-shot scores from a frozen vision-language model -- no
    human labels). Each concept's direction is defined *from the batch* as the
    difference of soft-weighted class means,

        d_k = normalize( mean_{s_k}(z)  -  mean_{1-s_k}(z) ),

    which is differentiable in the embedding z, so both losses reshape the
    encoder rather than just fitting a readout:

    - identification: BCE( a_k * <z, d_k> + b_k  ->  s_k ), i.e. the projection
      onto d_k must predict the concept (a_k, b_k are a learned per-concept
      logistic calibration).
    - orthogonality: mean of squared cosines <d_j, d_k> over penalized pairs.
      By default all pairs are penalized; call `set_pair_mask` to exempt
      concept pairs whose labels are correlated in the data.

    Exponential-moving-average class means are kept as buffers so that a
    concept with no batch support (rare concepts, small batches) still has a
    stable direction for the Gram matrix instead of dropping out of the loss.
    """

    def __init__(self, n_concepts: int, dim: int, ema: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.n, self.dim, self.ema, self.eps = n_concepts, dim, ema, eps
        # per-concept logistic calibration for the identification BCE
        self.a = nn.Parameter(torch.ones(n_concepts))
        self.b = nn.Parameter(torch.zeros(n_concepts))
        # EMA class means (detached): fallback directions for unsupported concepts
        self.register_buffer("mu_pos", torch.zeros(n_concepts, dim))
        self.register_buffer("mu_neg", torch.zeros(n_concepts, dim))
        self.register_buffer("inited", torch.zeros(n_concepts, dtype=torch.bool))

    @torch.no_grad()
    def _update_ema(self, k, mp, mn):
        if not self.inited[k]:
            self.mu_pos[k], self.mu_neg[k], self.inited[k] = mp, mn, True
        else:
            self.mu_pos[k] = self.ema * self.mu_pos[k] + (1 - self.ema) * mp
            self.mu_neg[k] = self.ema * self.mu_neg[k] + (1 - self.ema) * mn

    def set_pair_mask(self, mask: torch.Tensor):
        """Restrict the orthogonality penalty to selected concept pairs.

        mask: boolean [n, n]; True = penalize this pair's overlap. Pass
        `|corrcoef(labels)| < threshold` to exempt naturally-correlated pairs
        (conditional orthogonality). Unset = penalize every pair.
        """
        self.register_buffer("pair_mask", mask.bool(), persistent=False)

    def forward(self, Z, S):
        """Z: [B, dim] raw (unnormalized) image embeddings. S: [B, n] soft labels.

        Returns {"id": scalar, "orth": scalar, "D": [n, dim] detached unit
        concept directions for logging/eval}.
        """
        dirs, id_losses = [], []
        for k in range(self.n):
            s = S[:, k]
            wp, wn = s, 1.0 - s
            sp, sn = wp.sum(), wn.sum()
            if sp > 1e-3 and sn > 1e-3:
                # soft class means -> differentiable concept direction
                mp = (wp.unsqueeze(1) * Z).sum(0) / sp
                mn = (wn.unsqueeze(1) * Z).sum(0) / sn
                self._update_ema(k, mp.detach(), mn.detach())
                dhat = (mp - mn) / (mp - mn).norm().clamp(min=self.eps)
                id_losses.append(F.binary_cross_entropy_with_logits(
                    self.a[k] * (Z @ dhat) + self.b[k], s))
            elif self.inited[k]:
                # no batch support: EMA fallback keeps the direction defined
                d = self.mu_pos[k] - self.mu_neg[k]
                dhat = d / d.norm().clamp(min=self.eps)
            else:
                dhat = torch.zeros(self.dim, device=Z.device)
            dirs.append(dhat)
        D = torch.stack(dirs, 0)
        G = D @ D.t()                                            # pairwise cosines
        off = ~torch.eye(self.n, dtype=torch.bool, device=Z.device)
        if hasattr(self, "pair_mask"):
            off = off & self.pair_mask                           # conditional orth
        loss_id = torch.stack(id_losses).mean() if id_losses else Z.new_zeros(())
        return {"id": loss_id, "orth": (G[off] ** 2).mean(), "D": D.detach()}
