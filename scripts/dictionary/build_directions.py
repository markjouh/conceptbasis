"""Final concept directions: per-concept best-of-three construction, selected
on development CC0 attribute mentions, then relabel the full embedding cache.

Constructions compared per concept:
  generic      mean("an object that is <member>") - "an object"
  contrastive  mean(positive prompts) - mean(negative prompts)
  image        mu+ - mu- over VLM-verified anchor images (from verify_concepts)

Also flags degenerate concepts (universal: judge says yes to ~all proposed
negatives; vacuous: ~no proposed positives verified).

Outputs:
  data/concept_directions.npy    [n_concepts, dim] final unit directions
  data/direction_sources.json    {"source": per-concept choice, "flags": {...}}
  data/labels.parquet            GMM-calibrated soft labels + splits (relabeled)

By default the tracked per-concept source choices are reused exactly. Pass
``--selection reselect`` only when intentionally constructing a new set of
directions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from conceptbasis import BACKBONE as MODEL, BACKBONE_PRETRAINED as PRETRAINED
from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selection",
        choices=("frozen", "reselect"),
        default="frozen",
        help="reuse the tracked per-concept choices, or recompute them",
    )
    ap.add_argument(
        "--source-map",
        default="data/direction_sources.json",
        help="tracked choices used by --selection frozen",
    )
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--selection-attributes", default="data/attributes_dev.jsonl")
    ap.add_argument(
        "--selection-split",
        choices=("dev",),
        default="dev",
        help="held-out development split used to choose direction construction",
    )
    args = ap.parse_args()

    import open_clip
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    model.eval().to(dev)

    def temb(txts):
        out = []
        for i in range(0, len(txts), 128):
            t = tok(txts[i:i + 128]).to(dev)
            out.append(torch.nn.functional.normalize(model.encode_text(t), dim=-1).cpu().numpy())
        return np.concatenate(out)

    d = json.load(open(os.path.join(ROOT, "data/dictionary.json")))
    frozen_record = None
    if args.selection == "frozen":
        frozen_record = json.load(open(os.path.join(ROOT, args.source_map)))
        expected = {concept["name"] for concept in d}
        actual = set(frozen_record["source"])
        if actual != expected:
            raise ValueError(
                "frozen direction choices do not match the dictionary; "
                "run with --selection reselect to create new choices"
            )
    prompts = json.load(open(os.path.join(ROOT, "data/contrastive_prompts.json")))
    ids = json.load(open(os.path.join(ROOT, "data/image_ids.json")))
    id2i = {x: i for i, x in enumerate(ids)}
    fi = np.load(os.path.join(ROOT, "data/image_embeddings.npy"))
    labels_path = os.path.join(ROOT, "data/labels.parquet")
    df = pd.read_parquet(labels_path)
    if list(df.image_id) != ids:
        raise ValueError("label rows do not match image embedding order")
    train_ids = set(df.loc[df.split == "train", "image_id"])
    manifest = load_split_manifest(ROOT, args.split_manifest)

    # Development-only mention labels used to choose among direction recipes.
    rows = [
        json.loads(line)
        for line in open(os.path.join(ROOT, args.selection_attributes))
        if line.strip()
    ]
    rows = [r for r in rows if r.get("attributes")]
    cc0_order = json.load(open(os.path.join(ROOT, "data/cc0_image_ids.json")))
    cc0_index = {image_id: index for index, image_id in enumerate(cc0_order)}
    img_cc0_all = np.load(os.path.join(ROOT, "data/image_embeddings_cc0.npy"))
    if len(img_cc0_all) != len(cc0_order):
        raise ValueError("CC0 embeddings do not match their image ID manifest")
    if any(
        split_for_image(manifest, row["image_id"]) != args.selection_split
        for row in rows
    ):
        raise ValueError("selection attribute file contains a non-development class")
    sets = [set(r["attributes"]) for r in rows]
    img_cc0 = img_cc0_all[[cc0_index[row["image_id"]] for row in rows]]

    # verified anchors + degeneracy stats from the VLM judgments
    J = defaultdict(lambda: {"pos": [], "neg": []})
    stats = defaultdict(lambda: {"py": 0, "pn": 0, "ny": 0, "nn": 0})
    jpath = os.path.join(ROOT, "data/concept_judgments.jsonl")
    if os.path.exists(jpath):
        for l in open(jpath):
            r = json.loads(l)
            if r.get("verdict") is None:
                continue
            if r["image_id"] not in train_ids:
                continue
            st = stats[r["concept"]]
            if r["proposed"] == "pos":
                st["py" if r["verdict"] else "pn"] += 1
            else:
                st["ny" if r["verdict"] else "nn"] += 1
            if r["verdict"] and r["proposed"] == "pos" and r["image_id"] in id2i:
                J[r["concept"]]["pos"].append(id2i[r["image_id"]])
            if (not r["verdict"]) and r["proposed"] == "neg" and r["image_id"] in id2i:
                J[r["concept"]]["neg"].append(id2i[r["image_id"]])

    flags = {}
    for c in (c["name"] for c in d):
        st = stats[c]
        ny_rate = st["ny"] / max(1, st["ny"] + st["nn"])
        py_rate = st["py"] / max(1, st["py"] + st["pn"])
        if st["ny"] + st["nn"] >= 10 and ny_rate >= 0.75:
            flags[c] = "universal"
        elif st["py"] + st["pn"] >= 10 and py_rate <= 0.3:
            flags[c] = "vacuous"
    if frozen_record is not None:
        flags = frozen_record["flags"]

    e_base = temb(["an object"])[0]
    dim = fi.shape[1]
    final = np.zeros((len(d), dim), dtype=np.float32)
    src = {}
    aucs_final = []
    for k, c in enumerate(d):
        name = c["name"]
        cands = {}
        eo = temb([f"an object that is {m}" for m in c["members"]]).mean(0) - e_base
        cands["text-generic"] = eo / (np.linalg.norm(eo) + 1e-8)
        p = prompts.get(name, {})
        if p.get("positive") and p.get("negative"):
            en = temb(p["positive"]).mean(0) - temb(p["negative"]).mean(0)
            cands["text-contrastive"] = en / (np.linalg.norm(en) + 1e-8)
        pos, neg = J[name]["pos"], J[name]["neg"]
        image_validated = len(pos) >= 8 and len(neg) >= 8 and name not in flags
        if pos and neg and name not in flags:
            v = fi[pos].mean(0) - fi[neg].mean(0)
            cands["image-anchored"] = v / (np.linalg.norm(v) + 1e-8)

        mset = set(c["members"])
        y = np.array([1 if s & mset else 0 for s in sets])
        scores = None
        if y.sum() >= 10:
            score_candidates = {
                candidate_name: candidate
                for candidate_name, candidate in cands.items()
                if candidate_name != "image-anchored" or image_validated
            }
            scores = {
                candidate_name: roc_auc_score(y, img_cc0 @ candidate)
                for candidate_name, candidate in score_candidates.items()
            }

        if frozen_record is not None:
            recorded = frozen_record["source"][name]
            if recorded.startswith("FLAGGED_"):
                best = "text-generic"
            elif recorded in ("image", "image(unvalidated)"):
                best = "image-anchored"
            else:
                best = recorded
            if best not in cands:
                raise ValueError(f"frozen source {recorded!r} unavailable for {name!r}")
            src[name] = recorded
        elif name in flags:
            best = "text-generic"
            src[name] = "FLAGGED_" + flags[name]
        elif scores is not None:
            best = max(scores, key=scores.get)
            src[name] = best
        else:
            best = "image-anchored" if image_validated else "text-generic"
            src[name] = best
        final[k] = cands[best]
        if scores is not None:
            aucs_final.append(scores[best])

    np.save(os.path.join(ROOT, "data/concept_directions.npy"), final)
    provenance = {
        "selection_split": args.selection_split,
        "dictionary_sha256": sha256(os.path.join(ROOT, "data/dictionary.json")),
        "split_manifest_sha256": sha256(os.path.join(ROOT, args.split_manifest)),
        "selection_attributes_sha256": sha256(
            os.path.join(ROOT, args.selection_attributes)
        ),
        "judgments_sha256": sha256(jpath),
    }
    json.dump({"source": src, "flags": flags, "provenance": provenance},
              open(os.path.join(ROOT, "data/direction_sources.json"), "w"), indent=1)
    from collections import Counter
    print("construction choices:", dict(Counter(src.values())))
    if aucs_final:
        print(f"final direction AUROC vs mentions: mean {np.mean(aucs_final):.3f} "
              f"| p10 {np.percentile(aucs_final, 10):.3f}")

    # Relabel all partitions with calibrators fit on train classes only.
    S = fi @ final.T
    soft = np.zeros_like(S, dtype=np.float32)
    train = (df.split == "train").to_numpy()
    for k in range(S.shape[1]):
        values = S[:, k].reshape(-1, 1)
        gm = GaussianMixture(2, random_state=0, n_init=2).fit(values[train])
        soft[:, k] = gm.predict_proba(values)[:, int(gm.means_.argmax())]
    keep = [c for c in df.columns if not c.startswith("s_")]
    score_frame = pd.DataFrame(
        soft,
        columns=[f"s_{concept['name']}" for concept in d],
        index=df.index,
    )
    df = pd.concat([df[keep], score_frame], axis=1)
    df.to_parquet(labels_path)
    print(f"relabeled -> {labels_path}")


if __name__ == "__main__":
    main()
