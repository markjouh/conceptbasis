"""Stage 7 inspection — Render the finalized dictionary and tagged exemplars.

Example images are tag-backed: an image is eligible only when its mined open
tags contain the concept name or one of its merged member phrases. There
is deliberately no embedding-retrieval fallback.
"""
from __future__ import annotations
import argparse
import json
import os

from conceptbasis.site import public_nav, thumbnail_data_url
from conceptbasis.splits import image_class, load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def diverse_examples(candidates, topk):
    """Prefer one explicitly tagged exemplar per object class."""
    selected = []
    seen_classes = set()
    for candidate in candidates:
        concept = image_class(candidate[0]["image_id"])
        if concept not in seen_classes:
            selected.append(candidate)
            seen_classes.add(concept)
            if len(selected) == topk:
                return selected
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) == topk:
                break
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--dictionary",
        "--dict",
        dest="dictionary",
        default="data/dictionary_usage_profile_v8.json",
    )
    ap.add_argument(
        "--img-dir", nargs="+", default=["data/raw/object_images_CC0"]
    )
    ap.add_argument("--out", default="docs/dictionary.html")
    ap.add_argument("--title", default="Concept dictionary — 256 axes")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument(
        "--open-tags", "--attributes", dest="open_tags", nargs="+", default=None
    )
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument(
        "--gallery-split",
        choices=("train", "dev", "test", "non-test", "all"),
        default="dev",
    )
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.gallery_split in {"test", "all"} and not args.allow_test:
        raise ValueError("rendering test or all splits requires --allow-test")
    if args.open_tags is None:
        if args.gallery_split == "dev":
            args.open_tags = ["data/attributes_dev_vllm_gemma4_nvfp4.jsonl"]
        elif args.gallery_split == "train":
            args.open_tags = ["data/attributes_train.jsonl"]
        elif args.gallery_split == "non-test":
            args.open_tags = [
                "data/attributes_train_vllm_gemma4_nvfp4_open_tags_nonredundant_v8.jsonl",
                "data/attributes_dev_vllm_gemma4_nvfp4.jsonl",
            ]
        else:
            args.open_tags = ["data/heldout/attributes_test.jsonl"]

    path_by_id = {}
    for image_dir_arg in args.img_dir:
        image_dir = os.path.join(ROOT, image_dir_arg)
        for current, _, files in os.walk(image_dir):
            for filename in files:
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                path = os.path.join(current, filename)
                image_id = os.path.relpath(path, image_dir).replace(os.sep, "/")
                if image_id in path_by_id and path_by_id[image_id] != path:
                    raise ValueError(f"duplicate image ID across --img-dir: {image_id}")
                path_by_id[image_id] = path
    d = json.load(open(os.path.join(ROOT, args.dictionary)))
    manifest = load_split_manifest(ROOT, args.split_manifest)
    rows = []
    for attributes_arg in args.open_tags:
        rows.extend(
            json.loads(line)
            for line in open(os.path.join(ROOT, attributes_arg))
            if line.strip()
        )
    splits = {row["image_id"]: split_for_image(manifest, row["image_id"]) for row in rows}
    if args.gallery_split in {"train", "dev", "test"}:
        rows = [
            row
            for row in rows
            if splits[row["image_id"]] == args.gallery_split
        ]
    elif args.gallery_split == "non-test":
        rows = [row for row in rows if splits[row["image_id"]] != "test"]
    if not args.allow_test and any(splits[row["image_id"]] == "test" for row in rows):
        raise ValueError("attribute inputs contain test rows; pass --allow-test explicitly")
    image_ids = [row["image_id"] for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("attribute inputs contain duplicate image IDs")
    missing = [row["image_id"] for row in rows if row["image_id"] not in path_by_id]
    if missing:
        raise ValueError(f"attribute images are missing from --img-dir: {missing[:5]}")

    def matched_candidates(terms, canonical_name):
        terms = set(terms)
        candidates = []
        for row in rows:
            matched = terms.intersection(row.get("attributes", []))
            if matched:
                candidates.append((row, sorted(matched)))
        candidates.sort(
            key=lambda item: (
                canonical_name not in item[1],
                -len(item[1]),
                item[0]["image_id"],
            )
        )
        return diverse_examples(candidates, args.topk)

    def rendered_images(candidates):
        return [
            {
                "b": thumbnail_data_url(path_by_id[row["image_id"]], size=140),
                "n": os.path.splitext(os.path.basename(row["image_id"]))[0].replace("_", " "),
                "matched": ", ".join(matched),
            }
            for row, matched in candidates
        ]

    cards = []
    for c in d:
        negative_name = c.get("negative_name")
        positive_candidates = matched_candidates(c["members"], c["name"])
        negative_candidates = matched_candidates(
            c.get("negative_pole", []), negative_name or ""
        )
        cards.append({
            "name": c["name"], "prev": round(c["prevalence"] * 100, 1),
            "members": [m for m in c["members"] if m != c["name"]],
            "negative_name": negative_name,
            "negative_prev": (
                round(c["negative_prevalence"] * 100, 1)
                if "negative_prevalence" in c else None
            ),
            "neg": c.get("negative_pole", []),
            "pos_imgs": rendered_images(positive_candidates),
            "neg_imgs": rendered_images(negative_candidates),
        })

    scope_label = {
        "train": "train",
        "dev": "development",
        "test": "test",
        "non-test": "train + development",
        "all": "all supplied",
    }[args.gallery_split]
    html = (
        HTML.replace("__DATA__", json.dumps(cards))
        .replace("__NAV__", public_nav("dictionary"))
        .replace("__SCOPE__", scope_label)
        .replace("__TITLE__", args.title)
    )
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    covered = sum(bool(card["pos_imgs"] or card["neg_imgs"]) for card in cards)
    shown = sum(len(card["pos_imgs"]) + len(card["neg_imgs"]) for card in cards)
    print(
        f"wrote {out}  ({len(cards)} cards, {covered} with labeled examples, "
        f"{shown} images, {os.path.getsize(out)/1e6:.1f} MB)"
    )


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Concept dictionary</title>
<style>
 body{margin:0;background:#15171c;color:#e6e6e6;font:13px system-ui,sans-serif;padding:18px}
 h1{font-size:17px;margin:0 0 4px} .sub{color:#788;margin-bottom:14px}
 #q{background:#1d2027;border:1px solid #333a47;border-radius:8px;color:#e6e6e6;
    padding:8px 12px;width:320px;font-size:14px;margin-bottom:16px}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
 .card{background:#1c1f25;border:1px solid #2a2d34;border-radius:10px;padding:10px}
 .hd{display:flex;justify-content:space-between;align-items:baseline}
 .nm{font-weight:600;font-size:14px}
 .pv{color:#7fd1ff;font-size:12px}
 .mem{color:#889;font-size:11px;margin:3px 0 8px;line-height:1.35}
 .pos{color:#7fd1ff}.neg{color:#ff63b6}.pole{margin-top:7px}
 .polehd{font-size:10px;font-weight:700;letter-spacing:.08em;margin-bottom:4px}
 .ims{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}
 .ims div{text-align:center}
 .ims img{width:100%;border-radius:6px;background:#fff}
 .ims span{color:#667;font-size:9px;display:block;margin-top:2px;overflow:hidden;
           text-overflow:ellipsis;white-space:nowrap}
 .empty{color:#667;font-size:11px;padding:18px 0}
</style></head><body>__NAV__
<h1>__TITLE__</h1>
<div class="sub">examples are class-diverse __SCOPE__ images explicitly tagged with the concept name or a merged phrase</div>
<input id="q" placeholder="filter concepts… (name or member)" oninput="render(this.value)">
<div id="grid"></div>
<script>
const D=__DATA__;
function render(q){
 q=(q||'').toLowerCase();
 const g=document.getElementById('grid');g.innerHTML='';
 for(const c of D){
  if(q && !(c.name.toLowerCase().includes(q)||c.members.some(m=>m.toLowerCase().includes(q))||c.neg.some(m=>m.toLowerCase().includes(q))))continue;
  const el=document.createElement('div');el.className='card';
  const title=c.negative_name?`${c.name} ↔ ${c.negative_name}`:c.name;
  const prevalence=c.negative_prev===null?`${c.prev}%`:`${c.prev}% / ${c.negative_prev}%`;
  const positiveTerms=[c.name,...c.members].join(', ');
  const negativeTerms=c.neg.join(', ');
  const gallery=(label,klass,imgs)=>imgs.length
    ? `<div class="pole"><div class="polehd ${klass}">${label}</div><div class="ims">${imgs.map(i=>`<div><img src="${i.b}" title="matched tag: ${i.matched}"><span>${i.n} · ${i.matched}</span></div>`).join('')}</div></div>`
    : '';
  el.innerHTML=`<div class="hd"><span class="nm">${title}</span><span class="pv">${prevalence}</span></div>
   <div class="mem"><span class="pos">POS:</span> ${positiveTerms}${negativeTerms?`<br><span class="neg">NEG:</span> ${negativeTerms}`:''}</div>
   ${gallery('POSITIVE EXAMPLES','pos',c.pos_imgs)}
   ${gallery('NEGATIVE EXAMPLES','neg',c.neg_imgs)}
   ${(!c.pos_imgs.length&&!c.neg_imgs.length)?`<div class="empty">no explicitly tagged example in the __SCOPE__ pool</div>`:''}`;
  g.appendChild(el);
 }
}
render('');
</script></body></html>"""


if __name__ == "__main__":
    main()
