"""Stage 7 public output — Build the matched frozen-SigLIP2 control playground.

Sliders are z-scored projections onto the accepted dictionary's initial
member-averaged text directions.  The UI and gallery match the trained page,
so the two pages form a reproducible A/B comparison.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from make_trained_playground import (  # shared UI + selected artifact defaults
    HTML,
    SELECTED_DEV_LABELS,
    SELECTED_DICTIONARY,
    SELECTED_INPUTS,
)

from conceptbasis.site import public_nav, thumbnail_data_url
from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--soft-labels",
        "--labels",
        dest="soft_labels",
        default=f"{SELECTED_INPUTS}/labels.parquet",
    )
    ap.add_argument("--image-ids", default=f"{SELECTED_INPUTS}/image_ids.json")
    ap.add_argument("--image-embeddings", default=f"{SELECTED_INPUTS}/image_embeddings.npy")
    ap.add_argument("--cc0-embeddings", default=f"{SELECTED_INPUTS}/image_embeddings_cc0.npy")
    ap.add_argument("--directions", default=f"{SELECTED_INPUTS}/concept_directions_initial.npy")
    ap.add_argument("--dictionary", default=SELECTED_DICTIONARY)
    ap.add_argument("--cc0-image-ids", default="data/cc0_image_ids.json")
    ap.add_argument("--cc0-labels", default=SELECTED_DEV_LABELS)
    ap.add_argument("--gallery", type=int, default=1200)
    ap.add_argument("--out", default="docs/playground-baseline.html")
    ap.add_argument("--cc0", action="store_true",
                    help="public gallery from the freely-licensed CC0 subset")
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--gallery-split", choices=("dev", "test"), default="dev")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.gallery_split == "test" and not args.allow_test:
        raise ValueError("rendering test requires --allow-test")

    ids = json.load(open(os.path.join(ROOT, args.image_ids)))
    fi = np.load(os.path.join(ROOT, args.image_embeddings))
    D = np.load(os.path.join(ROOT, args.directions))
    df = pd.read_parquet(os.path.join(ROOT, args.soft_labels))
    scols = [c for c in df.columns if c.startswith("s_")]
    names = [c[2:] for c in scols]
    S = df[scols].to_numpy(dtype=np.float32)
    dictionary = json.load(open(os.path.join(ROOT, args.dictionary)))
    if names != [entry["name"] for entry in dictionary]:
        raise ValueError("soft-label and dictionary concept order differs")
    if D.shape != (len(names), fi.shape[1]):
        raise ValueError("initial concept directions have the wrong shape")
    if len(ids) != len(fi) or len(df) != len(fi):
        raise ValueError("image IDs, embeddings, and labels must have equal row counts")
    flags = {}
    tr = (df.split == "train").to_numpy()
    gallery_mask = (df.split == args.gallery_split).to_numpy()

    P_tr = fi[tr] @ D.T
    mu, sd = P_tr.mean(0), P_tr.std(0) + 1e-6
    gal = []
    if args.cc0:
        import re
        rows = [json.loads(l) for l in open(os.path.join(ROOT, args.cc0_labels)) if l.strip()]
        manifest = load_split_manifest(ROOT, args.split_manifest)
        fi_cc0 = np.load(os.path.join(ROOT, args.cc0_embeddings))
        if any(
            split_for_image(manifest, row["image_id"]) != args.gallery_split
            for row in rows
        ):
            raise ValueError("CC0 attribute file contains rows outside gallery split")
        cc0_order = json.load(open(os.path.join(ROOT, args.cc0_image_ids)))
        if len(fi_cc0) != len(cc0_order):
            raise ValueError("CC0 embeddings do not match their image ID manifest")
        cc0_index = {image_id: index for index, image_id in enumerate(cc0_order)}
        by_id = {row["image_id"]: row for row in rows if row.get("status") == "ok"}
        cc0_ids = [
            image_id for image_id in cc0_order
            if split_for_image(manifest, image_id) == args.gallery_split
        ]
        if set(by_id) != set(cc0_ids):
            raise ValueError("fixed-label file does not exactly cover the gallery split")
        fi_cc0 = fi_cc0[[cc0_index[image_id] for image_id in cc0_ids]]
        P_gal = (fi_cc0 @ D.T - mu) / sd
        cc0_dir = os.path.join(ROOT, "data/raw/object_images_CC0")
        for j, fn in enumerate(cc0_ids):
            present = set(by_id[fn]["present"])
            gal.append({"img": thumbnail_data_url(os.path.join(cc0_dir, fn), size=110, quality=72),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": re.sub(r"\d+$", "", os.path.splitext(fn)[0]).replace("_", " "),
                        "top": [names[k] for k in np.argsort(-P_gal[j]) if names[k] in present][:3]})
    else:
        pick = np.random.default_rng(1).permutation(np.where(gallery_mask)[0])[:args.gallery]
        P_gal = (fi[pick] @ D.T - mu) / sd
        for j, i in enumerate(pick):
            top = [names[k] for k in np.argsort(-S[i])[:3]]
            gal.append({"img": thumbnail_data_url(os.path.join(IMG_DIR, ids[i]), size=110, quality=72),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": ids[i].split(os.sep)[0].replace("_", " "),
                        "top": top})

    html = (HTML.replace("concept-direction sliders", "FROZEN control — text-direction profiles")
                .replace("__DIRECTION_LABEL__", "frozen backbone, no adapter")
                .replace("__MODEL_LABEL__", "SigLIP2 Giant · frozen member-averaged text-direction control")
                .replace("Green theme = trained adapter.", "Amber theme = control.")
                .replace("#6be0a8", "#ffb86b").replace("#2fb77f", "#e0a06b")
                .replace("#12171c", "#1a1512").replace("#182027", "#241d18")
                .replace("#24303a", "#3a2d24").replace("#2b3a44", "#473a2d")
                .replace("#22303a", "#342a22").replace("#33444f", "#483a2f")
                .replace("#2b3d49", "#433628"))
    data = {"names": names, "flags": flags, "gallery": gal,
            "meta": {"encoder": "siglip2-giant", "checkpoint": "frozen-text-control"}}
    html = html.replace("__DATA__", json.dumps(data))
    if args.cc0:
        html = html.replace("<body>", "<body>" + public_nav("baseline"))
    out = os.path.join(ROOT, args.out)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(gal)} gallery, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
