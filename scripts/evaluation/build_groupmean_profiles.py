"""Build standardized CC0 concept profiles for frozen or adapted embeddings."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def l2_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def profile(z: np.ndarray, z_cc0: np.ndarray, scores: np.ndarray,
            train: np.ndarray) -> np.ndarray:
    dirs = []
    for k in range(scores.shape[1]):
        s = scores[train, k]
        mp = (s[:, None] * z[train]).sum(0) / max(float(s.sum()), 1e-3)
        mn = ((1 - s)[:, None] * z[train]).sum(0) / max(float((1 - s).sum()), 1e-3)
        dirs.append(mp - mn)
    directions = l2_rows(np.stack(dirs))
    train_projection = z[train] @ directions.T
    mean = train_projection.mean(0)
    std = train_projection.std(0) + 1e-6
    return ((z_cc0 @ directions.T - mean) / std).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--cc0-embeddings", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--attributes", default="data/attributes.jsonl")
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--include-frozen", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    embeddings = np.load(os.path.join(ROOT, args.embeddings)).astype(np.float32)
    cc0 = np.load(os.path.join(ROOT, args.cc0_embeddings)).astype(np.float32)
    labels = pd.read_parquet(os.path.join(ROOT, args.labels))
    score_columns = [c for c in labels if c.startswith("s_")]
    names = np.array([c[2:] for c in score_columns])
    scores = labels[score_columns].to_numpy(dtype=np.float32)
    train = (labels.split == "train").to_numpy()
    rows = [json.loads(line) for line in open(os.path.join(ROOT, args.attributes)) if line.strip()]
    image_ids = np.array([r["image_id"] for r in rows])
    if len(cc0) != len(image_ids):
        raise ValueError("CC0 embedding rows do not match attribute rows")

    arrays = {"names": names, "image_ids": image_ids}
    if args.include_frozen:
        arrays["frozen"] = profile(embeddings, cc0, scores, train)
        print("built frozen", arrays["frozen"].shape, flush=True)

    from conceptbasis.train import Adapter
    for spec in args.checkpoint:
        if "=" not in spec:
            raise ValueError("--checkpoint must be NAME=PATH")
        name, path = spec.split("=", 1)
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        adapter = Adapter(embeddings.shape[1], ck["config"]["embed_dim"])
        adapter.load_state_dict(ck["img_adapter"])
        adapter.eval()
        with torch.no_grad():
            z = adapter(torch.from_numpy(embeddings)).numpy()
            z_cc0 = adapter(torch.from_numpy(cc0)).numpy()
        arrays[name] = profile(z, z_cc0, scores, train)
        print("built", name, arrays[name].shape, flush=True)

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays)
    print("wrote", out)


if __name__ == "__main__":
    main()
