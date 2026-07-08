"""Visual review page for the concept dictionary: one card per concept with
name, prevalence, member phrases, and the top-scoring THINGS images (CLIP
zero-shot on the mean of member-phrase embeddings). Search box to filter."""
from __future__ import annotations
import argparse
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAV_PUBLIC = '<div id="sitenav"><a href="index.html">⌂ Concept Basis</a><a href="playground.html">Playground</a><a href="playground-axis-aligned.html">Axis-aligned</a><a href="playground-baseline.html">Baseline</a><a href="dictionary.html" class=here>Dictionary</a><a href="attributes.html">Attributes</a></div>\n<style>#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.94);\nborder:1px solid #38404c;border-radius:20px;padding:6px 14px;font:12px system-ui;display:flex;gap:14px}\n#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover{color:#fff}\n#sitenav a.here{color:#fff;font-weight:600}</style>'
IMG_CACHE = os.path.join(ROOT, "data", "dictionary_image_embeddings.npy")


def thumb(path, size=140):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="data/dictionary.json")
    ap.add_argument("--img-dir", default="data/raw/object_images_CC0")
    ap.add_argument("--out", default="docs/dictionary.html")
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    img_dir = os.path.join(ROOT, args.img_dir)
    paths = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    img = np.load(IMG_CACHE)
    assert len(paths) == img.shape[0], (len(paths), img.shape)

    d = json.load(open(os.path.join(ROOT, args.dict)))
    import open_clip
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-256", pretrained="webli")
    tok = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")
    model.eval().to(dev)

    # neutral base for direction scoring (minimal-pair style: concept - base
    # cancels the generic-object component that dominates raw CLIP scores)
    tb = tok(["an object"]).to(dev)
    e_base = torch.nn.functional.normalize(model.encode_text(tb), dim=-1)[0].cpu().numpy()

    cards = []
    for c in d:
        t = tok([f"an object that is {m}" for m in c["members"]]).to(dev)
        e = torch.nn.functional.normalize(model.encode_text(t), dim=-1).mean(0).cpu().numpy()
        dvec = e - e_base
        dvec /= np.linalg.norm(dvec) + 1e-8
        scores = img @ dvec
        top = np.argsort(-scores)[:args.topk]
        cards.append({
            "name": c["name"], "prev": round(c["prevalence"] * 100, 1),
            "members": [m for m in c["members"] if m != c["name"]],
            "neg": c.get("negative_pole", []),
            "imgs": [{"b": thumb(paths[i]),
                      "n": os.path.splitext(os.path.basename(paths[i]))[0].replace("_", " ")}
                     for i in top],
        })

    html = HTML.replace("__DATA__", json.dumps(cards))
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(cards)} cards, {os.path.getsize(out)/1e6:.1f} MB)")


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
 .ims{display:flex;gap:6px}
 .ims div{text-align:center;flex:1}
 .ims img{width:100%;border-radius:6px;background:#fff}
 .ims span{color:#667;font-size:9px;display:block;margin-top:2px;overflow:hidden;
           text-overflow:ellipsis;white-space:nowrap}
</style></head><body>
<h1>Concept dictionary — 256 axes</h1>
<div class="sub">each card: concept, prevalence, merged phrases, top-4 CLIP-scoring THINGS images</div>
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
   <div class="ims">${c.imgs.map(i=>`<div><img src="data:image/jpeg;base64,${i.b}"><span>${i.n}</span></div>`).join('')}</div>`;
  g.appendChild(el);
 }
}
render('');
</script></body></html>"""


if __name__ == "__main__":
    main()
