"""THINGS concept-slider playground from the trained adapter.

256 sliders = the concept axes after post-hoc alignment rotation (orthonormalized
mu+ - mu- directions -> first 256 coords; residual 64 dims preserved when you
load a real object). Live nearest-neighbor retrieval over the test gallery.
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAV_PUBLIC = '<div id="sitenav"><a href="index.html">⌂ Concept Basis</a><a href="playground.html">Playground</a><a href="playground-axis-aligned.html" class=here>Axis-aligned</a><a href="playground-baseline.html">Baseline</a><a href="dictionary.html">Dictionary</a><a href="attributes.html">Attributes</a></div>\n<style>#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.94);\nborder:1px solid #38404c;border-radius:20px;padding:6px 14px;font:12px system-ui;display:flex;gap:14px}\n#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover{color:#fff}\n#sitenav a.here{color:#fff;font-weight:600}</style>'
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def thumb(path, size=110):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def alignment_rotation(D):
    k, d = D.shape
    U, _, Vt = np.linalg.svd(D, full_matrices=False)
    A = U @ Vt
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(np.concatenate([A.T, rng.standard_normal((d, d - k))], axis=1))
    R = Q.T
    R[:k] = A
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="outputs/checkpoints/latest")
    ap.add_argument("--gallery", type=int, default=1200)
    ap.add_argument("--out", default="outputs/playground_axis_aligned.html")
    ap.add_argument("--cc0", action="store_true",
                    help="public gallery from the freely-licensed CC0 subset")
    args = ap.parse_args()

    from conceptbasis.train import Adapter

    ck = torch.load(os.path.join(ROOT, args.run_dir, "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    d_out = cfg["embed_dim"]

    ids = json.load(open(os.path.join(ROOT, "data/image_ids.json")))
    fi = np.load(os.path.join(ROOT, "data/image_embeddings.npy"))
    df = pd.read_parquet(os.path.join(ROOT, os.environ.get("LABELS", "data/labels.parquet")))
    scols = [c for c in df.columns if c.startswith("s_")]
    names = [c[2:] for c in scols]
    S = df[scols].to_numpy(dtype=np.float32)

    img_ad = Adapter(fi.shape[1], d_out)
    img_ad.load_state_dict(ck["img_adapter"])
    img_ad.eval()
    with torch.no_grad():
        Z = img_ad(torch.from_numpy(fi)).numpy()

    tr = (df.split == "train").to_numpy()
    te = (df.split == "test").to_numpy()

    # soft mu+ - mu- directions on train -> alignment rotation, sign-oriented
    dirs = []
    for k in range(len(names)):
        s = S[tr, k]
        mp = (s[:, None] * Z[tr]).sum(0) / max(s.sum(), 1e-3)
        mn = ((1 - s)[:, None] * Z[tr]).sum(0) / max((1 - s).sum(), 1e-3)
        dirs.append(l2(mp - mn))
    D = np.stack(dirs)
    R = alignment_rotation(D)
    At = Z[tr] @ R.T
    for k in range(len(names)):
        if np.corrcoef(At[:, k], S[tr, k])[0, 1] < 0:
            R[k] = -R[k]
    At = Z[tr] @ R.T
    mean, std = At.mean(0), At.std(0) + 1e-6

    gal = []
    n_c = len(names)
    if args.cc0:
        import re
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/attributes.jsonl")) if l.strip()]
        cc0_ids = [r["image_id"] for r in rows if r.get("attributes")]
        fi_cc0 = np.load(os.path.join(ROOT, "data/image_embeddings_cc0.npy"))
        with torch.no_grad():
            A_cc0 = img_ad(torch.from_numpy(fi_cc0)).numpy() @ R.T
        cc0_dir = os.path.join(ROOT, "data/raw/object_images_CC0")
        for j, fn in enumerate(cc0_ids):
            z = (A_cc0[j, :n_c] - mean[:n_c]) / std[:n_c]
            gal.append({"img": thumb(os.path.join(cc0_dir, fn)),
                        "emb": [round(float(x), 3) for x in A_cc0[j]],
                        "concept": re.sub(r"\d+$", "", os.path.splitext(fn)[0]).replace("_", " "),
                        "top": [names[k] for k in np.argsort(-z)[:3]]})
    else:
        te_idx = np.where(te)[0]
        rng = np.random.default_rng(1)
        pick = rng.permutation(te_idx)[:args.gallery]
        A_all = Z @ R.T
        for i in pick:
            p = os.path.join(IMG_DIR, ids[i])
            top = [names[k] for k in np.argsort(-S[i])[:3]]
            gal.append({"img": thumb(p),
                        "emb": [round(float(x), 3) for x in A_all[i]],
                        "concept": ids[i].split(os.sep)[0].replace("_", " "),
                        "top": top})

    # slider order: by dictionary prevalence (matches dict order in labels file)
    data = {"names": names, "mean": [round(float(x), 3) for x in mean],
            "std": [round(float(x), 3) for x in std], "d": d_out, "gallery": gal}
    html = HTML.replace("__DATA__", json.dumps(data))
    if args.cc0:
        html = html.replace("<body>", "<body>" + NAV_PUBLIC)
    out = os.path.join(ROOT, args.out)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(gal)} gallery, {os.path.getsize(out)/1e6:.1f} MB)")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>THINGS sliders — axis-aligned view</title>
<style>
 body{margin:0;font:13px system-ui,sans-serif;background:#15171c;color:#e6e6e6;display:flex;height:100vh}
 #left{width:400px;overflow-y:auto;padding:12px;border-right:1px solid #2a2d34;display:flex;flex-direction:column}
 #right{flex:1;overflow-y:auto;padding:14px}
 h2{font-size:15px;margin:4px 0 8px}
 #q,#pick{background:#1d2027;border:1px solid #333a47;border-radius:8px;color:#e6e6e6;padding:7px 10px;margin:6px 0}
 #pick{border-color:#2f81f7}
 .btns{display:flex;gap:8px;margin:6px 0}
 button{background:#2a2d34;color:#e6e6e6;border:1px solid #3a3f48;border-radius:6px;padding:6px 10px;cursor:pointer}
 button:hover{background:#343943}
 #sliders{overflow-y:auto;flex:1}
 .row{display:flex;align-items:center;gap:8px;margin:2px 0}
 .row label{width:150px;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .row input{flex:1}
 .row .v{width:34px;text-align:right;color:#9aa;font-variant-numeric:tabular-nums;font-size:11px}
 .row.active label{color:#7fd1ff;font-weight:600}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:10px}
 .card{background:#1c1f25;border:1px solid #2a2d34;border-radius:8px;padding:6px;text-align:center}
 .card img{width:100%;border-radius:4px}
 .card .n{color:#7fd1ff;font-size:11px;font-weight:600;text-transform:capitalize;margin-top:3px}
 .card .t{color:#778;font-size:9px;line-height:1.25;margin-top:1px}
 .hint{color:#889;font-size:11px;margin:0 0 10px}
</style></head><body>
<div id="left">
 <h2>256 sliders — axis-aligned view</h2>
 <div class="btns"><button onclick="reset()">Reset</button><button onclick="randomObj()">Random object</button></div>
 <input id="pick" list="items" placeholder="load an item… (type its name)">
 <datalist id="items"></datalist>
 <div id="loaded" class="hint"></div>
 <input id="q" placeholder="filter sliders… (e.g. metallic, furry)" oninput="filter(this.value)">
 <div id="sliders"></div>
</div>
<div id="right">
 <h2>Nearest objects (live)</h2>
 <div class="hint">Reset = build from scratch. Random object = load a real object, then move one slider —
 its residual dims (unnamed identity) stay fixed. Non-zero sliders are highlighted.</div>
 <div id="grid"></div>
</div>
<script>
const DATA=__DATA__;
const N=DATA.names.length, D=DATA.d, G=DATA.gallery;
function norm(v){let s=0;for(const x of v)s+=x*x;s=Math.sqrt(s)||1;return v.map(x=>x/s);}
const GN=G.map(g=>norm(g.emb));
let base=DATA.mean.slice();
const sl=[];
const box=document.getElementById('sliders');
for(let k=0;k<N;k++){
 const row=document.createElement('div');row.className='row';row.dataset.name=DATA.names[k];
 const lab=document.createElement('label');lab.textContent=DATA.names[k];lab.title=DATA.names[k];
 const inp=document.createElement('input');inp.type='range';inp.min=-3;inp.max=3;inp.step=0.1;inp.value=0;
 const val=document.createElement('span');val.className='v';val.textContent='0.0';
 inp.oninput=()=>{val.textContent=(+inp.value).toFixed(1);row.classList.toggle('active',+inp.value!==0);update();};
 row.append(lab,inp,val);box.appendChild(row);sl.push({inp,val,row});
}
function filter(q){q=q.toLowerCase();
 for(const s of sl)s.row.style.display=s.row.dataset.name.includes(q)?'':'none';}
// absolute semantics: slider value v means coord = mean + v*std. Residual dims
// (256..d) always come from `base` (loaded item's unnamed identity, or mean).
function target(){
 const t=base.slice();
 for(let k=0;k<N;k++){const v=+sl[k].inp.value;t[k]=DATA.mean[k]+v*DATA.std[k];}
 return t;}
function setSlider(k,v){v=Math.max(-3,Math.min(3,v));
 sl[k].inp.value=v;sl[k].val.textContent=v.toFixed(1);
 sl[k].row.classList.toggle('active',Math.abs(v)>0.75);}
function loadItem(i){const g=G[i];base=g.emb.slice();
 for(let k=0;k<N;k++)setSlider(k,(g.emb[k]-DATA.mean[k])/DATA.std[k]);
 document.getElementById('loaded').textContent='loaded: '+g.concept+' — sliders show its concept profile';
 update();}
// populate the item picker (name #n, sorted)
(function(){
 const dl=document.getElementById('items');
 const cnt={};
 G.forEach((g,i)=>{cnt[g.concept]=(cnt[g.concept]||0)+1;g._label=g.concept+' #'+cnt[g.concept];});
 [...G.keys()].sort((a,b)=>G[a]._label.localeCompare(G[b]._label)).forEach(i=>{
  const o=document.createElement('option');o.value=G[i]._label;dl.appendChild(o);});
 document.getElementById('pick').addEventListener('change',e=>{
  const i=G.findIndex(g=>g._label===e.target.value);
  if(i>=0)loadItem(i);});
})();
function update(){
 const t=norm(target());
 const scored=GN.map((g,i)=>{let s=0;for(let k=0;k<D;k++)s+=t[k]*g[k];return[s,i];});
 scored.sort((a,b)=>b[0]-a[0]);
 const grid=document.getElementById('grid');grid.innerHTML='';
 for(let r=0;r<28;r++){const[s,i]=scored[r];const g=G[i];
  const c=document.createElement('div');c.className='card';
  c.innerHTML=`<img src="data:image/jpeg;base64,${g.img}"><div class="n">${g.concept}</div>
   <div class="t">${s.toFixed(3)} · ${g.top.join(', ')}</div>`;
  grid.appendChild(c);}
}
function reset(){base=DATA.mean.slice();
 for(let k=0;k<N;k++)setSlider(k,0);
 document.getElementById('loaded').textContent='';
 document.getElementById('pick').value='';
 update();}
function randomObj(){loadItem(Math.floor(Math.random()*G.length));}
update();
</script></body></html>"""


if __name__ == "__main__":
    main()
