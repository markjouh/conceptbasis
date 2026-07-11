"""Rebuild GMM-calibrated concept labels from cached embeddings and directions."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--template-labels", required=True,
                    help="Parquet supplying image_id, concept, split, and concept-column order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    embeddings = np.load(os.path.join(ROOT, args.embeddings))
    directions = np.load(os.path.join(ROOT, args.directions))
    template = pd.read_parquet(os.path.join(ROOT, args.template_labels))
    score_columns = [c for c in template.columns if c.startswith("s_")]
    if embeddings.shape[0] != len(template):
        raise ValueError("embedding rows do not match template-label rows")
    if directions.shape[0] != len(score_columns):
        raise ValueError("direction rows do not match template concept columns")

    scores = embeddings @ directions.T
    soft = np.empty_like(scores, dtype=np.float32)
    for k in range(scores.shape[1]):
        values = scores[:, k:k + 1]
        gm = GaussianMixture(2, random_state=0, n_init=2).fit(values)
        soft[:, k] = gm.predict_proba(values)[:, int(gm.means_.argmax())]
        if k % 32 == 0:
            print(f"calibrated {k}/{scores.shape[1]}", flush=True)

    out = template[["image_id", "concept", "split"]].copy()
    for k, column in enumerate(score_columns):
        out[column] = soft[:, k]
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path)
    print(json.dumps({
        "out": out_path,
        "shape": list(out.shape),
        "mean": float(soft.mean()),
        "std": float(soft.std()),
        "positive_rate": float((soft >= 0.5).mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
