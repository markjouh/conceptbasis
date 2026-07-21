"""Stage 7 inspection — Render images beside their mined open-set VLM tags.

The self-contained HTML page reviews dictionary-discovery annotations. It is
separate from the exhaustive TRUE/FALSE fixed-label page used after the
dictionary is frozen.
"""
from __future__ import annotations
import argparse
import json
import os

from conceptbasis.site import public_nav, thumbnail_data_url
from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--open-tags",
        "--attrs",
        dest="open_tags",
        default="data/attributes_dev_vllm_gemma4_nvfp4.jsonl",
    )
    ap.add_argument("--img-dir", default="data/raw/object_images_CC0")
    ap.add_argument("--out", default="docs/attributes.html")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--gallery-split", choices=("train", "dev", "test"), default="dev")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--layout", choices=("slides", "grid"), default="slides")
    args = ap.parse_args()
    if args.gallery_split == "test" and not args.allow_test:
        raise ValueError("rendering test requires --allow-test")

    rows = [
        json.loads(line)
        for line in open(os.path.join(ROOT, args.open_tags))
        if line.strip()
    ]
    manifest = load_split_manifest(ROOT, args.split_manifest)
    rows = [
        row for row in rows
        if row.get("attributes")
        and split_for_image(manifest, row["image_id"]) == args.gallery_split
    ][:args.limit]
    items = []
    for r in rows:
        p = os.path.join(ROOT, args.img_dir, r["image_id"])
        if not os.path.exists(p):
            continue
        items.append({"name": os.path.splitext(r["image_id"])[0],
                      "img": thumbnail_data_url(p, size=256), "attrs": r["attributes"]})
    data = json.dumps(items)
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    out = os.path.join(ROOT, args.out)
    with open(out, "w") as f:
        template = GRID_HTML if args.layout == "grid" else HTML
        f.write(template.replace("__DATA__", data).replace("__NAV__", public_nav("attributes")))
    print(f"wrote {out}  ({len(items)} items, {os.path.getsize(out)/1e6:.1f} MB)")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>THINGS open-set tags</title>
<style>
 body{margin:0;background:#15171c;color:#e6e6e6;font:14px system-ui,sans-serif;
      height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center}
 #counter{position:fixed;top:14px;right:18px;color:#889;font-variant-numeric:tabular-nums}
 #name{font-size:22px;font-weight:600;margin:6px 0 12px;text-transform:capitalize}
 #img{max-height:52vh;max-width:80vw;border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,.5);background:#fff}
 #attrs{display:flex;flex-wrap:wrap;gap:7px;justify-content:center;max-width:760px;margin:16px 0}
 .chip{background:#232732;border:1px solid #333a47;border-radius:20px;padding:5px 12px;color:#cfe3ff;font-size:13px}
 #nav{margin-top:8px;color:#667}
 button{background:#2a2d34;color:#e6e6e6;border:1px solid #3a3f48;border-radius:8px;
        padding:9px 16px;cursor:pointer;font-size:15px;margin:0 6px}
 button:hover{background:#343943}
 .hint{color:#556;font-size:12px;margin-top:10px}
</style></head><body>__NAV__
<div id="counter"></div>
<div id="name"></div>
<img id="img">
<div id="attrs"></div>
<div id="nav"><button onclick="go(-1)">&larr; Prev</button><button onclick="go(1)">Next &rarr;</button></div>
<div class="hint">use &larr; / &rarr; arrow keys</div>
<script>
const D=__DATA__; let i=0;
function render(){
 const it=D[i];
 document.getElementById('name').textContent=it.name.replace(/_/g,' ');
 document.getElementById('img').src=it.img;
 document.getElementById('counter').textContent=(i+1)+' / '+D.length;
 const a=document.getElementById('attrs');a.innerHTML='';
 for(const x of it.attrs){const c=document.createElement('span');c.className='chip';c.textContent=x;a.appendChild(c);}
}
function go(d){i=(i+d+D.length)%D.length;render();}
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(1);else if(e.key==='ArrowLeft')go(-1);});
render();
</script></body></html>"""


GRID_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>THINGS open-set tags · grid</title>
<style>
 body{margin:0;background:#15171c;color:#e6e6e6;font:14px system-ui,sans-serif}
 main{max-width:1500px;margin:72px auto 40px;padding:0 18px}
 h1{font-size:24px;margin:0 0 8px}.sub{color:#8e97a8;margin-bottom:22px}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}
 .card{background:#20242b;border:1px solid #343b46;border-radius:12px;padding:11px;min-width:0}
 .name{font-size:16px;font-weight:650;text-transform:capitalize;margin:0 0 9px;color:#8dc8ff}
 img{display:block;width:100%;height:220px;object-fit:contain;border-radius:8px;background:#fff}
 .attrs{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}
 .chip{background:#303746;border:1px solid #414b5d;border-radius:14px;padding:3px 8px;color:#d6e7ff;font-size:12px}
 @media(max-width:620px){#grid{grid-template-columns:1fr}img{height:280px}}
</style></head><body>__NAV__<main>
<h1>One-image-per-class open-set tags</h1><div class="sub" id="summary"></div><div id="grid"></div>
</main><script>
const D=__DATA__; document.getElementById('summary').textContent=D.length+' tagged images';
const grid=document.getElementById('grid');
for(const it of D){
 const card=document.createElement('section');card.className='card';
 const name=document.createElement('div');name.className='name';name.textContent=it.name.replace(/_/g,' ');
 const img=document.createElement('img');img.src=it.img;img.alt=it.name;
 const attrs=document.createElement('div');attrs.className='attrs';
 for(const x of it.attrs){const chip=document.createElement('span');chip.className='chip';chip.textContent=x;attrs.appendChild(chip);}
 card.append(name,img,attrs);grid.appendChild(card);
}
</script></body></html>"""


if __name__ == "__main__":
    main()
