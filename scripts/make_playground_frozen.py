"""Control playground (frozen SigLIP2, no adapter): sliders = the object's
z-scored projection profile onto the HYBRID TEXT concept directions. Same UI
and semantics as v5 (load item -> sliders show its profile; scoring = slider
vector vs gallery profiles), so trained-vs-frozen is a fair A/B."""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from make_playground_directions import HTML, thumb  # shared UI + thumbnailer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAV_PUBLIC = '<div id="sitenav"><a href="index.html">⌂ Concept Basis</a><a href="playground.html">Playground</a><a href="playground-axis-aligned.html">Axis-aligned</a><a href="playground-baseline.html" class=here>Baseline</a><a href="dictionary.html">Dictionary</a><a href="attributes.html">Attributes</a></div>\n<style>#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.94);\nborder:1px solid #38404c;border-radius:20px;padding:6px 14px;font:12px system-ui;display:flex;gap:14px}\n#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover{color:#fff}\n#sitenav a.here{color:#fff;font-weight:600}</style>'
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", type=int, default=1200)
    ap.add_argument("--out", default="outputs/playground_frozen.html")
    ap.add_argument("--cc0", action="store_true",
                    help="public gallery from the freely-licensed CC0 subset")
    args = ap.parse_args()

    ids = json.load(open(os.path.join(ROOT, "data/image_ids.json")))
    fi = np.load(os.path.join(ROOT, "data/image_embeddings.npy"))
    D = np.load(os.path.join(ROOT, "data/concept_directions.npy"))
    df = pd.read_parquet(os.path.join(ROOT, "data/labels.parquet"))
    scols = [c for c in df.columns if c.startswith("s_")]
    names = [c[2:] for c in scols]
    S = df[scols].to_numpy(dtype=np.float32)
    flags = json.load(open(os.path.join(ROOT, "data/direction_sources.json")))["flags"]
    tr = (df.split == "train").to_numpy()
    te = (df.split == "test").to_numpy()

    P_tr = fi[tr] @ D.T
    mu, sd = P_tr.mean(0), P_tr.std(0) + 1e-6
    gal = []
    if args.cc0:
        import re
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/attributes.jsonl")) if l.strip()]
        cc0_ids = [r["image_id"] for r in rows if r.get("attributes")]
        fi_cc0 = np.load(os.path.join(ROOT, "data/image_embeddings_cc0.npy"))
        P_gal = (fi_cc0 @ D.T - mu) / sd
        cc0_dir = os.path.join(ROOT, "data/raw/object_images_CC0")
        for j, fn in enumerate(cc0_ids):
            gal.append({"img": thumb(os.path.join(cc0_dir, fn)),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": re.sub(r"\d+$", "", os.path.splitext(fn)[0]).replace("_", " "),
                        "top": [names[k] for k in np.argsort(-P_gal[j])[:3]]})
    else:
        pick = np.random.default_rng(1).permutation(np.where(te)[0])[:args.gallery]
        P_gal = (fi[pick] @ D.T - mu) / sd
        for j, i in enumerate(pick):
            top = [names[k] for k in np.argsort(-S[i])[:3]]
            gal.append({"img": thumb(os.path.join(IMG_DIR, ids[i])),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": ids[i].split(os.sep)[0].replace("_", " "),
                        "top": top})

    html = (HTML.replace("concept-direction sliders", "FROZEN control — text-direction profiles")
                .replace("Nearest objects (model's true concept directions)",
                         "Nearest objects (frozen SigLIP2, no adapter)")
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
