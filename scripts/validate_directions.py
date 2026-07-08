"""Old vs new direction construction, validated per concept vs VLM mentions (CC0).

old: mean('an object that is <member>') - 'an object'
new: mean(contrastive positives) - mean(contrastive negatives)

Writes data/concept_directions_contrastive.npy and prints the comparison table.
"""
from __future__ import annotations
import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL, PRETRAINED = "ViT-B-16-SigLIP2-256", "webli"


@torch.no_grad()
def main():
    import open_clip
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    model.eval().to(dev)

    def temb(txts):
        out = []
        for i in range(0, len(txts), 256):
            t = tok(txts[i:i + 256]).to(dev)
            out.append(torch.nn.functional.normalize(model.encode_text(t), dim=-1).cpu().numpy())
        return np.concatenate(out)

    d = json.load(open(os.path.join(ROOT, "data/dictionary.json")))
    prompts = json.load(open(os.path.join(ROOT, "data/contrastive_prompts.json")))

    # CC0 mention labels (weak ground truth)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/attributes.jsonl")) if l.strip()]
    rows = [r for r in rows if r.get("attributes")]
    cc0_ids = [r["image_id"] for r in rows]
    sets = [set(r["attributes"]) for r in rows]

    # CC0 image embeddings with THIS backbone (cache)
    cache = os.path.join(ROOT, "data", "image_embeddings_cc0.npy")
    if os.path.exists(cache):
        img = np.load(cache)
    else:
        from PIL import Image
        paths = [os.path.join(ROOT, "data/raw/object_images_CC0", x) for x in cc0_ids]
        out = []
        for i in range(0, len(paths), 128):
            ims = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + 128]]).to(dev)
            out.append(torch.nn.functional.normalize(model.encode_image(ims), dim=-1).cpu().numpy())
        img = np.concatenate(out)
        np.save(cache, img)

    e_base = temb(["an object"])[0]
    D_old, D_new = [], []
    for c in d:
        eo = temb([f"an object that is {m}" for m in c["members"]]).mean(0) - e_base
        D_old.append(eo / (np.linalg.norm(eo) + 1e-8))
        p = prompts[c["name"]]
        en = temb(p["positive"]).mean(0) - temb(p["negative"]).mean(0)
        D_new.append(en / (np.linalg.norm(en) + 1e-8))
    D_old, D_new = np.stack(D_old), np.stack(D_new)
    np.save(os.path.join(ROOT, "data/concept_directions_contrastive.npy"), D_new.astype(np.float32))

    print(f"{'concept':28s} {'old':>6s} {'new':>6s}  (AUROC vs mentions, >=10 pos)")
    deltas, table = [], []
    for k, c in enumerate(d):
        mset = set(c["members"])
        y = np.array([1 if s & mset else 0 for s in sets])
        if y.sum() < 10:
            continue
        a_old = roc_auc_score(y, img @ D_old[k])
        a_new = roc_auc_score(y, img @ D_new[k])
        deltas.append(a_new - a_old)
        table.append((c["name"], a_old, a_new))
    table.sort(key=lambda r: r[2] - r[1])
    print("biggest losses:")
    for nm, ao, an in table[:8]:
        print(f"  {nm:28s} {ao:.3f} {an:.3f}  ({an-ao:+.3f})")
    print("biggest gains:")
    for nm, ao, an in table[-12:][::-1]:
        print(f"  {nm:28s} {ao:.3f} {an:.3f}  ({an-ao:+.3f})")
    d_arr = np.array(deltas)
    olds = np.array([r[1] for r in table]); news = np.array([r[2] for r in table])
    print(f"\nmean AUROC: old {olds.mean():.3f} -> new {news.mean():.3f}  "
          f"(improved {int((d_arr>0.01).sum())}, regressed {int((d_arr<-0.01).sum())}, n={len(d_arr)})")


if __name__ == "__main__":
    main()
