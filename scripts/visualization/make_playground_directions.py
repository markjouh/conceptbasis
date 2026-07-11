"""Default playground: sliders act on the model's TRUE mu+ - mu- concept directions
(trained adapter space). No orthonormalization rotation, no cosine-vs-gallery-norm
division — score(g) = w_anchor * cos(z_g, z_base) + sum_k v_k * zscore_k(g),
where zscore_k(g) = (z_g . d_k - mean_k) / std_k.

Trade: axes are the model's honest (correlated) directions — pushing 'manmade'
may honestly co-move its family. Extremes should now match the 0%-intruder
ceiling measured for the raw directions.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAV_PUBLIC = '<div id="sitenav"><a href="index.html">⌂ Concept Basis</a><a href="playground.html" class=here>Playground</a><a href="playground-baseline.html">Baseline</a><a href="dictionary.html">Dictionary</a><a href="attributes.html">Attributes</a></div>\n<style>#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.94);\nborder:1px solid #38404c;border-radius:20px;padding:6px 14px;font:12px system-ui;display:flex;gap:14px}\n#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover{color:#fff}\n#sitenav a.here{color:#fff;font-weight:600}</style>'
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def thumb(path, size=110):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="outputs/checkpoints/latest")
    ap.add_argument("--labels", default="data/labels.parquet")
    ap.add_argument("--gallery", type=int, default=1200)
    ap.add_argument("--out", default="outputs/playground_directions.html")
    ap.add_argument("--cc0", action="store_true",
                    help="public gallery from the freely-licensed CC0 subset "
                         "(one image per THINGS concept; redistributable)")
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--gallery-split", choices=("dev", "test"), default="dev")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.gallery_split == "test" and not args.allow_test:
        raise ValueError("rendering test requires --allow-test")

    from conceptbasis.models import Adapter

    ck = torch.load(os.path.join(ROOT, args.run_dir, "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    ids = json.load(open(os.path.join(ROOT, "data/image_ids.json")))
    fi = np.load(os.path.join(ROOT, "data/image_embeddings.npy"))
    df = pd.read_parquet(os.path.join(ROOT, args.labels))
    scols = [c for c in df.columns if c.startswith("s_")]
    names = [c[2:] for c in scols]
    S = df[scols].to_numpy(dtype=np.float32)
    flags = json.load(open(os.path.join(ROOT, "data/direction_sources.json")))["flags"]

    img_ad = Adapter(fi.shape[1], ck["config"]["embed_dim"])
    img_ad.load_state_dict(ck["img_adapter"])
    img_ad.eval()
    with torch.no_grad():
        Z = img_ad(torch.from_numpy(fi)).numpy()

    tr = (df.split == "train").to_numpy()
    gallery_mask = (df.split == args.gallery_split).to_numpy()

    # true soft mu+ - mu- directions (train), per-axis projection stats
    dirs = []
    for k in range(len(names)):
        s = S[tr, k]
        mp = (s[:, None] * Z[tr]).sum(0) / max(s.sum(), 1e-3)
        mn = ((1 - s)[:, None] * Z[tr]).sum(0) / max((1 - s).sum(), 1e-3)
        dirs.append(l2(mp - mn))
    D = np.stack(dirs)
    P_tr = Z[tr] @ D.T
    mu, sd = P_tr.mean(0), P_tr.std(0) + 1e-6

    gal = []
    if args.cc0:
        # public gallery: CC0 subset (redistributable), one image per concept
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/attributes_dev.jsonl"))
                if l.strip()]
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
        with torch.no_grad():
            Z_cc0 = img_ad(torch.from_numpy(fi_cc0)).numpy()
        P_gal = (Z_cc0 @ D.T - mu) / sd
        cc0_dir = os.path.join(ROOT, "data/raw/object_images_CC0")
        import re
        for j, fn in enumerate(cc0_ids):
            top = [names[k] for k in np.argsort(-P_gal[j])[:3]]
            concept = re.sub(r"\d+$", "", os.path.splitext(fn)[0]).replace("_", " ")
            gal.append({"img": thumb(os.path.join(cc0_dir, fn)),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": concept, "top": top})
    else:
        gallery_idx = np.where(gallery_mask)[0]
        rng = np.random.default_rng(1)
        pick = rng.permutation(gallery_idx)[:args.gallery]
        P_gal = (Z[pick] @ D.T - mu) / sd                # z-scored projections
        for j, i in enumerate(pick):
            top = [names[k] for k in np.argsort(-S[i])[:3]]
            gal.append({"img": thumb(os.path.join(IMG_DIR, ids[i])),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": ids[i].split(os.sep)[0].replace("_", " "),
                        "top": top})

    order = list(np.argsort([-S[:, k].mean() for k in range(len(names))]))
    data = {"names": names, "flags": flags, "gallery": gal}
    html = HTML.replace("__DATA__", json.dumps(data))
    if args.cc0:
        html = html.replace("<body>", "<body>" + NAV_PUBLIC)
    out = os.path.join(ROOT, args.out)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(gal)} gallery, {os.path.getsize(out)/1e6:.1f} MB)")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>concept-direction sliders</title>
<style>
 body{margin:0;font:13px system-ui,sans-serif;background:#12171c;color:#e6e6e6;display:flex;height:100vh}
 #left{width:400px;overflow-y:auto;padding:12px;border-right:1px solid #24303a;display:flex;flex-direction:column}
 #right{flex:1;overflow-y:auto;padding:14px}
 h2{font-size:15px;margin:4px 0 8px;color:#6be0a8}
 #q,#pick{background:#182027;border:1px solid #2b3a44;border-radius:8px;color:#e6e6e6;padding:7px 10px;margin:6px 0}
 #pick{border-color:#2fb77f}
 .btns{display:flex;gap:8px;margin:6px 0}
 button{background:#22303a;color:#e6e6e6;border:1px solid #33444f;border-radius:6px;padding:6px 10px;cursor:pointer}
 button:hover{background:#2b3d49}
 #sliders{overflow-y:auto;flex:1}
 .row{display:flex;align-items:center;gap:8px;margin:2px 0}
 .row label{width:150px;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .row.flagged label{color:#666;text-decoration:line-through}
 .row input{flex:1}
 .row .v{width:34px;text-align:right;color:#9aa;font-size:11px}
 .row.active label{color:#6be0a8;font-weight:600}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:10px}
 .card{background:#182027;border:1px solid #24303a;border-radius:8px;padding:6px;text-align:center}
 .card img{width:100%;border-radius:4px}
 .card .n{color:#6be0a8;font-size:11px;font-weight:600;text-transform:capitalize;margin-top:3px}
 .card .t{color:#778;font-size:9px;line-height:1.25;margin-top:1px}
 .hint{color:#7a8a8a;font-size:11px;margin:0 0 10px}
</style></head><body>
<div id="left">
 <h2>concept-direction sliders</h2>
 <div class="btns"><button onclick="reset()">Reset</button><button onclick="randomObj()">Random object</button></div>
 <input id="pick" list="items" placeholder="anchor to an item… (type name)">
 <datalist id="items"></datalist>
 <div id="loaded" class="hint"></div>
 <input id="q" placeholder="filter sliders…" oninput="filter(this.value)">
 <div id="sliders"></div>
</div>
<div id="right">
 <h2>Nearest objects (model's true concept directions)</h2>
 <div class="hint">Sliders = the object's projection profile onto the model's true concept
 directions. Load an item to SEE its embedding on the sliders, then edit any axis and retrieve.
 Struck-through sliders = flagged degenerate axes. Green theme = default (true directions).</div>
 <div id="grid"></div>
</div>
<script>
const DATA=__DATA__;
const N=DATA.names.length, G=DATA.gallery;
const sl=[];
const box=document.getElementById('sliders');
for(let k=0;k<N;k++){
 const row=document.createElement('div');row.className='row';row.dataset.name=DATA.names[k];
 if(DATA.flags[DATA.names[k]])row.classList.add('flagged');
 const lab=document.createElement('label');lab.textContent=DATA.names[k];lab.title=DATA.names[k];
 const inp=document.createElement('input');inp.type='range';inp.min=-3;inp.max=3;inp.step=0.1;inp.value=0;
 const val=document.createElement('span');val.className='v';val.textContent='0.0';
 inp.oninput=()=>{val.textContent=(+inp.value).toFixed(1);row.classList.toggle('active',+inp.value!==0);update();};
 row.append(lab,inp,val);box.appendChild(row);sl.push({inp,val,row});
}
function filter(q){q=q.toLowerCase();
 for(const s of sl)s.row.style.display=s.row.dataset.name.includes(q)?'':'none';}
function setSlider(k,v){v=Math.max(-3,Math.min(3,v));
 sl[k].inp.value=v;sl[k].val.textContent=v.toFixed(1);
 sl[k].row.classList.toggle('active',Math.abs(v)>0.75);}
function update(){
 const w=sl.map(s=>+s.inp.value);
 const scored=G.map((g,i)=>{
  let s=0;
  for(let k=0;k<N;k++)if(w[k]!==0)s+=w[k]*g.p[k];
  return[s,i];});
 scored.sort((a,b)=>b[0]-a[0]);
 const grid=document.getElementById('grid');grid.innerHTML='';
 for(let r=0;r<28;r++){const[s,i]=scored[r];const g=G[i];
  const c=document.createElement('div');c.className='card';
  c.innerHTML=`<img src="data:image/jpeg;base64,${g.img}"><div class="n">${g.concept}</div>
   <div class="t">${s.toFixed(2)} · ${g.top.join(', ')}</div>`;
  grid.appendChild(c);}
}
function reset(){
 for(let k=0;k<N;k++)setSlider(k,0);
 document.getElementById('loaded').textContent='';document.getElementById('pick').value='';
 update();}
function loadItem(i){const g=G[i];
 for(let k=0;k<N;k++)setSlider(k,g.p[k]);
 document.getElementById('loaded').textContent='loaded: '+g.concept+' — sliders show its embedding profile (edit freely)';
 update();}
function randomObj(){loadItem(Math.floor(Math.random()*G.length));}
(function(){
 const dl=document.getElementById('items');const cnt={};
 G.forEach((g,i)=>{cnt[g.concept]=(cnt[g.concept]||0)+1;g._label=g.concept+' #'+cnt[g.concept];});
 [...G.keys()].sort((a,b)=>G[a]._label.localeCompare(G[b]._label)).forEach(i=>{
  const o=document.createElement('option');o.value=G[i]._label;dl.appendChild(o);});
 document.getElementById('pick').addEventListener('change',e=>{
  const i=G.findIndex(g=>g._label===e.target.value);
  if(i>=0)loadItem(i);});
})();
update();
</script></body></html>"""


if __name__ == "__main__":
    main()
