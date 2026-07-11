"""Visual review page for the concept dictionary.

Example images are label-backed: an image is eligible only when its generated
attributes contain the concept name or one of its merged member phrases. There
is deliberately no embedding-retrieval fallback.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os

from PIL import Image

from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAV_PUBLIC = '<div id="sitenav"><a href="index.html">⌂ Concept Basis</a><a href="playground.html">Playground</a><a href="playground-baseline.html">Baseline</a><a href="dictionary.html" class=here>Dictionary</a><a href="attributes.html">Attributes</a></div>\n<style>#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.94);\nborder:1px solid #38404c;border-radius:20px;padding:6px 14px;font:12px system-ui;display:flex;gap:14px}\n#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover{color:#fff}\n#sitenav a.here{color:#fff;font-weight:600}</style>'
def thumb(path, size=140):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="data/dictionary.json")
    ap.add_argument("--img-dir", default="data/raw/object_images_CC0")
    ap.add_argument("--out", default="docs/dictionary.html")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--attributes", default=None)
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--gallery-split", choices=("dev", "test"), default="dev")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.gallery_split == "test" and not args.allow_test:
        raise ValueError("rendering test requires --allow-test")
    if args.attributes is None:
        args.attributes = (
            "data/attributes_dev.jsonl"
            if args.gallery_split == "dev"
            else "data/heldout/attributes_test.jsonl"
        )

    img_dir = os.path.join(ROOT, args.img_dir)
    paths = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    path_by_id = {os.path.basename(path): path for path in paths}
    d = json.load(open(os.path.join(ROOT, args.dict)))
    manifest = load_split_manifest(ROOT, args.split_manifest)
    rows = [
        json.loads(line)
        for line in open(os.path.join(ROOT, args.attributes))
        if line.strip()
    ]
    if any(
        split_for_image(manifest, row["image_id"]) != args.gallery_split
        for row in rows
    ):
        raise ValueError("attribute file contains rows outside --gallery-split")
    missing = [row["image_id"] for row in rows if row["image_id"] not in path_by_id]
    if missing:
        raise ValueError(f"attribute images are missing from --img-dir: {missing[:5]}")

    cards = []
    for c in d:
        members = set(c["members"])
        candidates = []
        for row in rows:
            matched = members.intersection(row.get("attributes", []))
            if matched:
                candidates.append((row, sorted(matched)))
        candidates.sort(
            key=lambda item: (
                c["name"] not in item[1],
                -len(item[1]),
                item[0]["image_id"],
            )
        )
        cards.append({
            "name": c["name"], "prev": round(c["prevalence"] * 100, 1),
            "members": [m for m in c["members"] if m != c["name"]],
            "neg": c.get("negative_pole", []),
            "imgs": [
                {
                    "b": thumb(path_by_id[row["image_id"]]),
                    "n": os.path.splitext(os.path.basename(row["image_id"]))[0].replace("_", " "),
                    "matched": ", ".join(matched),
                }
                for row, matched in candidates[:args.topk]
            ],
        })

    html = HTML.replace("__DATA__", json.dumps(cards)).replace(
        "__SPLIT__", args.gallery_split
    )
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    covered = sum(bool(card["imgs"]) for card in cards)
    shown = sum(len(card["imgs"]) for card in cards)
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
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
 .card{background:#1c1f25;border:1px solid #2a2d34;border-radius:10px;padding:10px}
 .hd{display:flex;justify-content:space-between;align-items:baseline}
 .nm{font-weight:600;font-size:14px}
 .pv{color:#7fd1ff;font-size:12px}
 .mem{color:#889;font-size:11px;margin:3px 0 8px;line-height:1.35}
 .neg{color:#e08;font-size:11px}
 .ims{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}
 .ims div{text-align:center}
 .ims img{width:100%;border-radius:6px;background:#fff}
 .ims span{color:#667;font-size:9px;display:block;margin-top:2px;overflow:hidden;
           text-overflow:ellipsis;white-space:nowrap}
 .empty{color:#667;font-size:11px;padding:18px 0}
</style></head><body>
<h1>Concept dictionary — 256 axes</h1>
<div class="sub">examples are __SPLIT__ images explicitly tagged with the concept name or a merged phrase</div>
<input id="q" placeholder="filter concepts… (name or member)" oninput="render(this.value)">
<div id="grid"></div>
<script>
const D=__DATA__;
function render(q){
 q=(q||'').toLowerCase();
 const g=document.getElementById('grid');g.innerHTML='';
 for(const c of D){
  if(q && !(c.name.toLowerCase().includes(q)||c.members.some(m=>m.toLowerCase().includes(q))))continue;
  const el=document.createElement('div');el.className='card';
  let mem=c.members.length?('= '+c.members.join(', ')):'&nbsp;';
  if(c.neg.length)mem+=' <span class="neg">| NEG: '+c.neg.join(', ')+'</span>';
  el.innerHTML=`<div class="hd"><span class="nm">${c.name}</span><span class="pv">${c.prev}%</span></div>
   <div class="mem">${mem}</div>
   ${c.imgs.length
     ? `<div class="ims">${c.imgs.map(i=>`<div><img src="data:image/jpeg;base64,${i.b}" title="matched tag: ${i.matched}"><span>${i.n} · ${i.matched}</span></div>`).join('')}</div>`
     : `<div class="empty">no explicitly tagged example in the __SPLIT__ split</div>`}`;
  g.appendChild(el);
 }
}
render('');
</script></body></html>"""


if __name__ == "__main__":
    main()
