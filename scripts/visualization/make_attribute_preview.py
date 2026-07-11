"""Self-contained HTML viewer to flip through THINGS images + the visual
attributes the VLM generated for each. Opens in a browser (file://)."""
from __future__ import annotations
import argparse
import base64
import io
import json
import os

from PIL import Image

from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAV_PUBLIC = '<div id="sitenav"><a href="index.html">⌂ Concept Basis</a><a href="playground.html">Playground</a><a href="playground-baseline.html">Baseline</a><a href="dictionary.html">Dictionary</a><a href="attributes.html" class=here>Attributes</a></div>\n<style>#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.94);\nborder:1px solid #38404c;border-radius:20px;padding:6px 14px;font:12px system-ui;display:flex;gap:14px}\n#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover{color:#fff}\n#sitenav a.here{color:#fff;font-weight:600}</style>'


def thumb_b64(path: str, size: int = 420) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", default="data/attributes_dev.jsonl")
    ap.add_argument("--img-dir", default="data/raw/object_images_CC0")
    ap.add_argument("--out", default="docs/attributes.html")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--gallery-split", choices=("dev", "test"), default="dev")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.gallery_split == "test" and not args.allow_test:
        raise ValueError("rendering test requires --allow-test")

    rows = [json.loads(l) for l in open(os.path.join(ROOT, args.attrs)) if l.strip()]
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
                      "img": thumb_b64(p), "attrs": r["attributes"]})
    data = json.dumps(items)
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    out = os.path.join(ROOT, args.out)
    with open(out, "w") as f:
        f.write(HTML.replace("__DATA__", data))
    print(f"wrote {out}  ({len(items)} items, {os.path.getsize(out)/1e6:.1f} MB)")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>THINGS attributes</title>
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
</style></head><body>
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
 document.getElementById('img').src='data:image/jpeg;base64,'+it.img;
 document.getElementById('counter').textContent=(i+1)+' / '+D.length;
 const a=document.getElementById('attrs');a.innerHTML='';
 for(const x of it.attrs){const c=document.createElement('span');c.className='chip';c.textContent=x;a.appendChild(c);}
}
function go(d){i=(i+d+D.length)%D.length;render();}
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(1);else if(e.key==='ArrowLeft')go(-1);});
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
