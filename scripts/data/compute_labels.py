"""Dense soft labels + frozen features for the 26k THINGS training set.

Backbone: conceptbasis.BACKBONE (Perception Encoder Core G). For every image:
  - frozen image embedding (saved; doubles as the adapter's training features)
  - score against each of the 256 dictionary concept directions
    (mean member text emb - 'an object' base), GMM-calibrated to [0,1]

Outputs:
  data/image_embeddings.npy      [N, d] float32 (L2-normalized)
  data/image_ids.json         image_id order for the matrix
  data/labels.parquet        image_id, concept folder, split, s_<concept>...
  data/concept_directions_initial.npy  [256, d] prompt-based text directions
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.mixture import GaussianMixture

from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")
from conceptbasis import BACKBONE as MODEL, BACKBONE_PRETRAINED as PRETRAINED


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", default="data/splits.json")
    args = parser.parse_args()
    import open_clip
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    model.eval().to(dev)

    ids = []
    for cur, _, fs in os.walk(IMG_DIR):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                ids.append(os.path.relpath(os.path.join(cur, f), IMG_DIR))
    ids.sort()
    print(f"{len(ids)} images")

    # ---- image embeddings ----
    emb_path = os.path.join(ROOT, "data", "image_embeddings.npy")
    ids_path = os.path.join(ROOT, "data", "image_ids.json")
    if os.path.exists(emb_path):
        img = np.load(emb_path)
        cached_ids = json.load(open(ids_path))
        if cached_ids != ids or img.shape[0] != len(ids):
            raise ValueError("cached image embeddings do not match the current image order")
        print("loaded cached embeddings")
    else:
        out = []
        for i in range(0, len(ids), 128):
            ims = torch.stack([pre(Image.open(os.path.join(IMG_DIR, x)).convert("RGB"))
                               for x in ids[i:i + 128]]).to(dev)
            out.append(torch.nn.functional.normalize(model.encode_image(ims), dim=-1)
                       .cpu().numpy().astype(np.float32))
            if (i // 128) % 20 == 0:
                print(f"  embedded {i + len(out[-1])}/{len(ids)}", flush=True)
        img = np.concatenate(out)
        np.save(emb_path, img)
        json.dump(ids, open(ids_path, "w"))

    # ---- concept directions ----
    d = json.load(open(os.path.join(ROOT, "data", "dictionary.json")))
    e_base = torch.nn.functional.normalize(
        model.encode_text(tok(["an object"]).to(dev)), dim=-1)[0].cpu().numpy()
    dirs = []
    for c in d:
        t = tok([f"an object that is {m}" for m in c["members"]]).to(dev)
        e = torch.nn.functional.normalize(model.encode_text(t), dim=-1).mean(0).cpu().numpy()
        v = e - e_base
        dirs.append(v / (np.linalg.norm(v) + 1e-8))
    D = np.stack(dirs).astype(np.float32)
    np.save(os.path.join(ROOT, "data", "concept_directions_initial.npy"), D)

    manifest = load_split_manifest(ROOT, args.split_manifest)
    concept_of = [x.split(os.sep)[0] for x in ids]
    split = np.array([split_for_image(manifest, image_id) for image_id in ids])
    train = split == "train"

    # ---- scores -> train-fitted GMM-calibrated soft labels ----
    S = img @ D.T                                          # [N, 256]
    soft = np.zeros_like(S, dtype=np.float32)
    for k in range(S.shape[1]):
        all_scores = S[:, k].reshape(-1, 1)
        gm = GaussianMixture(2, random_state=0, n_init=2).fit(all_scores[train])
        soft[:, k] = gm.predict_proba(all_scores)[:, int(gm.means_.argmax())]
        if k % 32 == 0:
            print(f"  calibrated {k}/256", flush=True)

    out = pd.DataFrame({"image_id": ids, "concept": concept_of, "split": split})
    score_frame = pd.DataFrame(
        soft,
        columns=[f"s_{concept['name']}" for concept in d],
    )
    out = pd.concat([out, score_frame], axis=1)
    out.to_parquet(os.path.join(ROOT, "data", "labels.parquet"))
    print("split sizes:", out.split.value_counts().to_dict())
    print("mean soft-positive rate (s>=0.5):", float((soft >= 0.5).mean()))
    print("wrote data/labels.parquet")


if __name__ == "__main__":
    main()
