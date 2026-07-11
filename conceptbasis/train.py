"""Train the concept-basis adapter on frozen backbone features.

Both towers are precomputed (image embeddings from compute_labels.py;
caption embeddings computed+cached here on first run), so each epoch is a few
seconds of MLP math on MPS.

  z_img = img_adapter(f_img)   [B, d]   d = 256 concept + residual
  z_txt = txt_adapter(f_txt)   [B, d]

  L = clip(masked) + lambda_orth * orthogonality
      (orthogonality on soft mu+/mu- class-mean directions; correlation-aware
      pair weighting via --corr_exempt/--corr_weighting)

  python -m conceptbasis.train [--embed_dim 320 ...]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from conceptbasis.losses import ConceptOrthogonalityLoss, symmetric_clip_loss
from conceptbasis.models import Adapter
from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from conceptbasis import BACKBONE as MODEL, BACKBONE_PRETRAINED as PRETRAINED


@torch.no_grad()
def caption_embeddings(ids: list[str], cap_path: str, cache: str) -> np.ndarray:
    if os.path.exists(cache):
        return np.load(cache)
    import open_clip
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    model.eval().to(dev)
    caps = {}
    for line in open(cap_path):
        if line.strip():
            r = json.loads(line)
            if r.get("caption"):
                caps[r["image_id"]] = r["caption"]
    texts = [caps.get(i, "an object") for i in ids]
    out = []
    for i in range(0, len(texts), 512):
        t = tok(texts[i:i + 512]).to(dev)
        out.append(F.normalize(model.encode_text(t), dim=-1).cpu().numpy().astype(np.float32))
    emb = np.concatenate(out)
    np.save(cache, emb)
    return emb


@torch.no_grad()
def evaluate(img_ad, txt_ad, fi, ft, S, dev, n_concepts, batch=4096):
    img_ad.eval(); txt_ad.eval()
    zi = torch.cat([img_ad(fi[i:i + batch].to(dev)).cpu() for i in range(0, len(fi), batch)])
    zt = torch.cat([txt_ad(ft[i:i + batch].to(dev)).cpu() for i in range(0, len(ft), batch)])
    zin, ztn = F.normalize(zi, dim=-1), F.normalize(zt, dim=-1)

    # retrieval (captions ~unique): exact-match R@k, subsampled gallery for speed
    n = min(2000, len(zin))
    sim = zin[:n] @ ztn[:n].T
    order = sim.argsort(dim=1, descending=True)
    rank = (order == torch.arange(n).view(-1, 1)).float().argmax(1)
    r_at = {k: float((rank < k).float().mean()) for k in (1, 5, 10)}

    # concept AUROC + orthogonality via soft mu+/mu- on raw z
    Z, Snp = zi.numpy(), S.numpy()
    dirs, aucs = [], []
    for k in range(n_concepts):
        s = Snp[:, k]
        wp, wn = s, 1 - s
        mp = (wp[:, None] * Z).sum(0) / max(wp.sum(), 1e-3)
        mn = (wn[:, None] * Z).sum(0) / max(wn.sum(), 1e-3)
        v = mp - mn
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            v = v / nv
            dirs.append(v)
            hard = s >= 0.5
            if 0 < hard.sum() < len(hard):
                aucs.append(roc_auc_score(hard, Z @ v))
    D = np.stack(dirs)
    G = D @ D.T
    off = ~np.eye(len(D), dtype=bool)
    img_ad.train(); txt_ad.train()
    return {"R@k": r_at, "auroc": float(np.mean(aucs)),
            "orth_rms": float(np.sqrt((G[off] ** 2).mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed_dim", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_orth", type=float, default=8.0)
    ap.add_argument("--ema", type=float, default=0.9)
    ap.add_argument("--corr_exempt", type=float, default=0.15,
                    help="correlation scale/threshold; set 0 for unconditional orthogonality")
    ap.add_argument("--corr_weighting", choices=("hard", "smooth"), default="smooth",
                    help="hard threshold mask or smooth correlation weighting")
    ap.add_argument("--corr_weight_floor", type=float, default=0.01,
                    help="minimum pair weight in smooth mode")
    ap.add_argument("--corr_weight_power", type=float, default=4.0,
                    help="exponent in exp(-(|corr|/corr_exempt)^power)")
    ap.add_argument("--image_ids", default="data/image_ids.json")
    ap.add_argument("--image_embeddings", default="data/image_embeddings.npy")
    ap.add_argument("--caption_embeddings", default="data/caption_embeddings.npy")
    ap.add_argument("--captions", default="data/captions.jsonl")
    ap.add_argument("--labels", default=os.environ.get("LABELS", "data/labels.parquet"))
    ap.add_argument("--split_manifest", default="data/splits.json")
    ap.add_argument("--run_name", default="adapter_d320")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    dev = ("mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device={dev}", flush=True)

    ids = json.load(open(os.path.join(ROOT, args.image_ids)))
    fi_all = np.load(os.path.join(ROOT, args.image_embeddings))
    ft_all = caption_embeddings(ids, os.path.join(ROOT, args.captions),
                                os.path.join(ROOT, args.caption_embeddings))
    df = pd.read_parquet(os.path.join(ROOT, args.labels))
    assert list(df.image_id) == ids
    manifest = load_split_manifest(ROOT, args.split_manifest)
    expected_splits = [split_for_image(manifest, image_id) for image_id in ids]
    if list(df.split) != expected_splits:
        raise ValueError("labels do not follow the class-level split manifest")
    split_path = os.path.join(ROOT, args.split_manifest)
    with open(split_path, "rb") as file:
        args.split_manifest_sha256 = hashlib.sha256(file.read()).hexdigest()
    scols = [c for c in df.columns if c.startswith("s_")]
    S_all = df[scols].to_numpy(dtype=np.float32)
    n_concepts = len(scols)

    masks = {s: (df.split == s).to_numpy() for s in ("train", "dev", "test")}
    fi = {s: torch.from_numpy(fi_all[m]) for s, m in masks.items()}
    ft = {s: torch.from_numpy(ft_all[m]) for s, m in masks.items()}
    S = {s: torch.from_numpy(S_all[m]) for s, m in masks.items()}
    print({s: int(m.sum()) for s, m in masks.items()}, f"| {n_concepts} concepts | d={args.embed_dim}")

    img_ad = Adapter(fi_all.shape[1], args.embed_dim).to(dev)
    txt_ad = Adapter(ft_all.shape[1], args.embed_dim).to(dev)
    closs = ConceptOrthogonalityLoss(n_concepts, args.embed_dim, args.ema).to(dev)
    if args.corr_exempt > 0:
        C = np.corrcoef(S_all[masks["train"]].T)
        if args.corr_weighting == "hard":
            mask = torch.from_numpy(np.abs(C) < args.corr_exempt).to(dev)
            closs.set_pair_mask(mask)
            n_ex = int((~mask.cpu().numpy() & ~np.eye(n_concepts, dtype=bool)).sum() / 2)
            print(f"correlation-aware orth: exempting {n_ex} naturally-correlated pairs "
                  f"(|corr| >= {args.corr_exempt})")
        else:
            if not 0 <= args.corr_weight_floor < 1:
                raise ValueError("--corr_weight_floor must be in [0, 1)")
            if args.corr_weight_power <= 0:
                raise ValueError("--corr_weight_power must be positive")
            weights = args.corr_weight_floor + (1 - args.corr_weight_floor) * np.exp(
                -(np.abs(C) / args.corr_exempt) ** args.corr_weight_power)
            weights = weights.astype(np.float32)
            np.fill_diagonal(weights, 0.0)
            closs.set_pair_weights(torch.from_numpy(weights).to(dev))
            off = ~np.eye(n_concepts, dtype=bool)
            q = np.quantile(weights[off], [0, .1, .25, .5, .75, .9, 1])
            print("smooth correlation-aware orth: "
                  f"tau={args.corr_exempt} floor={args.corr_weight_floor} "
                  f"power={args.corr_weight_power} mean_weight={weights[off].mean():.4f} "
                  f"quantiles={np.round(q, 4).tolist()}")
    logit_scale = torch.nn.Parameter(
        torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=dev)
    )
    params = list(img_ad.parameters()) + list(txt_ad.parameters()) + \
        list(closs.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    run_dir = os.path.join(ROOT, "outputs", "checkpoints", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    ntr = len(fi["train"])
    history = []

    for ep in range(args.epochs):
        perm = torch.randperm(ntr)
        agg = {"clip": 0.0, "orth": 0.0, "n": 0}
        for i in range(0, ntr - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            x = fi["train"][idx].to(dev)
            t = ft["train"][idx].to(dev)
            y = S["train"][idx].to(dev)
            zi, zt = img_ad(x), txt_ad(t)
            l_clip = symmetric_clip_loss(
                F.normalize(zi, dim=-1),
                F.normalize(zt, dim=-1),
                logit_scale.clamp(max=np.log(100)).exp(),
            )
            cl = closs(zi, y)
            loss = l_clip + args.lambda_orth * cl["orth"]
            opt.zero_grad(); loss.backward(); opt.step()
            agg["clip"] += l_clip.item()
            agg["orth"] += cl["orth"].item(); agg["n"] += 1
        sched.step()
        n = agg["n"]
        if ep % 5 == 4 or ep == args.epochs - 1:
            ev = evaluate(img_ad, txt_ad, fi["dev"], ft["dev"], S["dev"], dev, n_concepts)
            history.append({"epoch": ep, "train": {k: agg[k] / n for k in ("clip", "orth")},
                            "dev": ev})
            print(f"ep{ep:03d} clip={agg['clip']/n:.3f} orth={agg['orth']/n:.4f} "
                  f"| dev R@1={ev['R@k'][1]:.3f} "
                  f"R@5={ev['R@k'][5]:.3f} auroc={ev['auroc']:.3f} "
                  f"orth_rms={ev['orth_rms']:.4f}", flush=True)

    torch.save({"img_adapter": img_ad.state_dict(), "txt_adapter": txt_ad.state_dict(),
                "orthogonality_loss": closs.state_dict(), "logit_scale": logit_scale.detach().cpu(),
                "config": vars(args)}, os.path.join(run_dir, "ckpt.pt"))
    json.dump(history, open(os.path.join(run_dir, "history.json"), "w"), indent=2)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
