"""Simple concept-dictionary builder in IMAGE-SCORE space.

1. CLIP-embed all images and all unique mined phrases.
2. Score matrix S[img, phrase]; standardize per phrase -> score profiles.
3. Agglomerative clustering on profile correlation, cut at exactly --k clusters.
   (synonyms correlate -> merge; antonyms ANTI-correlate -> stay apart)
4. Negation sweep: cluster pairs with profile corr < --neg-thresh merge into one
   signed axis (keep the more-mentioned name).

Output: data/dictionary.json
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_CACHE = os.path.join(ROOT, "data", "dictionary_image_embeddings.npy")


@torch.no_grad()
def clip_embed(img_dir: str, phrases: list[str]):
    import open_clip
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-256", pretrained="webli")
    tok = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")
    model.eval().to(dev)

    if os.path.exists(IMG_CACHE):
        img = np.load(IMG_CACHE)
    else:
        paths = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        out = []
        for i in range(0, len(paths), 256):
            ims = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + 256]]).to(dev)
            out.append(torch.nn.functional.normalize(model.encode_image(ims), dim=-1).cpu().numpy())
        img = np.concatenate(out)
        np.save(IMG_CACHE, img)

    txt = []
    for i in range(0, len(phrases), 256):
        t = tok([f"an object that is {p}" for p in phrases[i:i + 256]]).to(dev)
        txt.append(torch.nn.functional.normalize(model.encode_text(t), dim=-1).cpu().numpy())
    return img, np.concatenate(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", default="data/attributes.jsonl")
    ap.add_argument("--img-dir", default="data/raw/object_images_CC0")
    ap.add_argument("--out", default="data/dictionary.json")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--min-mentions", type=int, default=3,
                    help="drop one-off phrases before clustering")
    ap.add_argument("--neg-thresh", type=float, default=-0.5)
    ap.add_argument("--merge-corr", type=float, default=0.6,
                    help="min profile correlation for phrases to merge (complete linkage)")
    ap.add_argument("--max-twin-corr", type=float, default=0.75,
                    help="skip a candidate concept if its profile correlates above this with a selected one")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, args.attrs)) if l.strip()]
    img_sets = [set(r["attributes"]) for r in rows if r.get("attributes")]
    counts = Counter(a for s in img_sets for a in s)
    phrases = sorted(p for p, c in counts.items() if c >= args.min_mentions)
    print(f"{len(img_sets)} images | {len(counts)} unique phrases | {len(phrases)} with >= {args.min_mentions} mentions")

    # ---- lexical negation fold-in: 'inorganic' -> negative pole of 'organic'.
    # CLIP text embeddings are negation-blind (measured: organic/inorganic profile
    # corr +0.7 even after de-confounding), so this must be lexical.
    NEG = ("in", "un", "non", "anti", "dis")
    base_of = {}
    pset = set(phrases)
    for p in phrases:
        w = p.replace("-", "")
        for pre in NEG:
            if w.startswith(pre) and w[len(pre):] in pset:
                base_of[p] = w[len(pre):]
    phrases = [p for p in phrases if p not in base_of]
    print(f"negation fold-in: {len(base_of)} phrases -> negative poles ({dict(list(base_of.items())[:6])})")

    img, txt = clip_embed(os.path.join(ROOT, args.img_dir), phrases)
    S = img @ txt.T                                      # [n_img, n_phrase]
    S = (S - S.mean(0)) / (S.std(0) + 1e-8)              # standardized profiles
    # remove top-1 shared component ("generic object-ness") so vague properties
    # don't all inter-correlate and glom into attractor clusters
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    S = S - (U[:, :1] * sv[:1]) @ Vt[:1]
    n = len(phrases)

    # profile correlation -> distance; COMPLETE linkage + a strict threshold so
    # phrases only merge when every pair is similar (no chain/blob merges).
    # Cluster count is then organic; we keep the top-k by mention support.
    C = np.corrcoef(S.T)                                 # [n, n]
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="complete")
    labels = fcluster(Z, t=1.0 - args.merge_corr, criterion="distance")
    print(f"complete-linkage @ corr>{args.merge_corr} -> {len(set(labels))} clusters (target {args.k})")

    # build concepts (attach folded-in negations to their base's cluster)
    concepts = []
    for l in sorted(set(labels)):
        idxs = np.where(labels == l)[0]
        members = [phrases[i] for i in idxs]
        name = max(members, key=lambda m: counts[m])
        negs = sorted(p for p, b in base_of.items() if b in members)
        mset = set(members)
        prev = sum(1 for s in img_sets if s & mset) / len(img_sets)
        profile = S[:, idxs].mean(1)
        c = {"name": name, "members": members, "prevalence": prev,
             "mentions": sum(counts[m] for m in members), "profile": profile}
        if negs:
            c["negative_pole"] = negs
        concepts.append(c)

    # greedy support-ranked selection with a diversity guard: skip candidates
    # whose de-confounded profile correlates > --max-twin-corr with anything
    # already selected (no near-twin axes; freed slots go to distinct concepts)
    concepts.sort(key=lambda c: -c["mentions"])
    P = np.stack([c["profile"] for c in concepts])
    P = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-8)
    merged, skipped = [], []
    sel_P = []
    for ci, c in enumerate(concepts):
        if len(merged) >= args.k:
            break
        if sel_P:
            corr = np.stack(sel_P) @ P[ci] / P.shape[1]
            if corr.max() > args.max_twin_corr:
                skipped.append((c["name"], merged[int(corr.argmax())]["name"], float(corr.max())))
                continue
        merged.append(c)
        sel_P.append(P[ci])
    print(f"greedy diverse selection: kept {len(merged)}, skipped {len(skipped)} near-twins")
    for nm, tw, cv in skipped[:12]:
        print(f"   skipped {nm!r} (corr {cv:.2f} with {tw!r})")
    for c in merged:
        c.pop("profile")
    merged.sort(key=lambda c: -c["prevalence"])
    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {args.out} ({len(merged)} concepts)")

    signed = [c for c in merged if c.get("negative_pole")]
    print(f"\nsigned axes ({len(signed)}):")
    for c in signed[:15]:
        print(f"  {c['name']}  <-vs->  {', '.join(c['negative_pole'])}")


if __name__ == "__main__":
    main()
