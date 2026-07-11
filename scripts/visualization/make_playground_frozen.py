"""Control playground (frozen backbone, no adapter): sliders = the object's
z-scored projection profile onto the HYBRID TEXT concept directions. Same UI
and semantics as v5 (load item -> sliders show its profile; scoring = slider
vector vs gallery profiles), so trained-vs-frozen is a fair A/B."""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from make_playground_directions import HTML, thumb  # shared UI + thumbnailer

from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAV_PUBLIC = '<div id="sitenav"><a href="index.html">⌂ Concept Basis</a><a href="playground.html">Playground</a><a href="playground-baseline.html" class=here>Baseline</a><a href="dictionary.html">Dictionary</a><a href="attributes.html">Attributes</a></div>\n<style>#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.94);\nborder:1px solid #38404c;border-radius:20px;padding:6px 14px;font:12px system-ui;display:flex;gap:14px}\n#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover{color:#fff}\n#sitenav a.here{color:#fff;font-weight:600}</style>'
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", type=int, default=1200)
    ap.add_argument("--out", default="outputs/playground_frozen.html")
    ap.add_argument("--cc0", action="store_true",
                    help="public gallery from the freely-licensed CC0 subset")
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--gallery-split", choices=("dev", "test"), default="dev")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.gallery_split == "test" and not args.allow_test:
        raise ValueError("rendering test requires --allow-test")

    ids = json.load(open(os.path.join(ROOT, "data/image_ids.json")))
    fi = np.load(os.path.join(ROOT, "data/image_embeddings.npy"))
    D = np.load(os.path.join(ROOT, "data/concept_directions.npy"))
    df = pd.read_parquet(os.path.join(ROOT, "data/labels.parquet"))
    scols = [c for c in df.columns if c.startswith("s_")]
    names = [c[2:] for c in scols]
    S = df[scols].to_numpy(dtype=np.float32)
    flags = json.load(open(os.path.join(ROOT, "data/direction_sources.json")))["flags"]
    tr = (df.split == "train").to_numpy()
    gallery_mask = (df.split == args.gallery_split).to_numpy()

    P_tr = fi[tr] @ D.T
    mu, sd = P_tr.mean(0), P_tr.std(0) + 1e-6
    gal = []
    if args.cc0:
        import re
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/attributes_dev.jsonl")) if l.strip()]
        manifest = load_split_manifest(ROOT, args.split_manifest)
        fi_cc0 = np.load(os.path.join(ROOT, "data/image_embeddings_cc0.npy"))
        if any(
            split_for_image(manifest, row["image_id"]) != args.gallery_split
            for row in rows
        ):
            raise ValueError("CC0 attribute file contains rows outside gallery split")
        cc0_order = json.load(open(os.path.join(ROOT, "data/cc0_image_ids.json")))
        if len(fi_cc0) != len(cc0_order):
            raise ValueError("CC0 embeddings do not match their image ID manifest")
        cc0_index = {image_id: index for index, image_id in enumerate(cc0_order)}
        cc0_ids = [row["image_id"] for row in rows if row.get("attributes")]
        fi_cc0 = fi_cc0[[cc0_index[image_id] for image_id in cc0_ids]]
        P_gal = (fi_cc0 @ D.T - mu) / sd
        cc0_dir = os.path.join(ROOT, "data/raw/object_images_CC0")
        for j, fn in enumerate(cc0_ids):
            gal.append({"img": thumb(os.path.join(cc0_dir, fn)),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": re.sub(r"\d+$", "", os.path.splitext(fn)[0]).replace("_", " "),
                        "top": [names[k] for k in np.argsort(-P_gal[j])[:3]]})
    else:
        pick = np.random.default_rng(1).permutation(np.where(gallery_mask)[0])[:args.gallery]
        P_gal = (fi[pick] @ D.T - mu) / sd
        for j, i in enumerate(pick):
            top = [names[k] for k in np.argsort(-S[i])[:3]]
            gal.append({"img": thumb(os.path.join(IMG_DIR, ids[i])),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": ids[i].split(os.sep)[0].replace("_", " "),
                        "top": top})

    html = (HTML.replace("concept-direction sliders", "FROZEN control — text-direction profiles")
                .replace("Nearest objects (model's true concept directions)",
                         "Nearest objects (frozen backbone, no adapter)")
                .replace("Green theme = default (true directions).", "Amber theme = control.")
                .replace("#6be0a8", "#ffb86b").replace("#2fb77f", "#e0a06b")
                .replace("#12171c", "#1a1512").replace("#182027", "#241d18")
                .replace("#24303a", "#3a2d24").replace("#2b3a44", "#473a2d")
                .replace("#22303a", "#342a22").replace("#33444f", "#483a2f")
                .replace("#2b3d49", "#433628"))
    data = {"names": names, "flags": flags, "gallery": gal}
    html = html.replace("__DATA__", json.dumps(data))
    if args.cc0:
        html = html.replace("<body>", "<body>" + NAV_PUBLIC)
    out = os.path.join(ROOT, args.out)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(gal)} gallery, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
