"""Stage 7 public output — Build the trained-adapter concept playground.

No orthonormalization rotation or cosine-vs-gallery-norm division is applied:
``score(g) = sum_k v_k * zscore_k(g)``, where
``zscore_k(g) = (z_g . d_k - mean_k) / std_k`` and every ``v_k >= 0``.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from conceptbasis.site import public_nav, thumbnail_data_url
from conceptbasis.splits import load_split_manifest, split_for_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SELECTED_INPUTS = "outputs/training_inputs/siglip2-gopt-p16-384@ad3410b/usage-profile-v8-v1"
SELECTED_CHECKPOINT = "outputs/checkpoints/siglip2_giant_usage_profile_v8_v11_reverse_ridge_s0"
SELECTED_DICTIONARY = "data/dictionary_usage_profile_v8.json"
SELECTED_DEV_LABELS = (
    "data/dictionary_labels_cc0_dev_vllm_gemma4_nvfp4_"
    "usage_profile_v8_object_grounded_v11.jsonl"
)
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run-dir", "--run_dir", default=SELECTED_CHECKPOINT)
    ap.add_argument(
        "--soft-labels",
        "--labels",
        dest="soft_labels",
        default=f"{SELECTED_INPUTS}/labels.parquet",
    )
    ap.add_argument("--image-ids", default=f"{SELECTED_INPUTS}/image_ids.json")
    ap.add_argument("--image-embeddings", default=f"{SELECTED_INPUTS}/image_embeddings.npy")
    ap.add_argument("--cc0-embeddings", default=f"{SELECTED_INPUTS}/image_embeddings_cc0.npy")
    ap.add_argument("--cc0-image-ids", default="data/cc0_image_ids.json")
    ap.add_argument("--cc0-labels", default=SELECTED_DEV_LABELS)
    ap.add_argument("--dictionary", default=SELECTED_DICTIONARY)
    ap.add_argument("--flags", default=None)
    ap.add_argument("--gallery", type=int, default=1200)
    ap.add_argument("--out", default="docs/playground.html")
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
    ids = json.load(open(os.path.join(ROOT, args.image_ids)))
    fi = np.load(os.path.join(ROOT, args.image_embeddings))
    df = pd.read_parquet(os.path.join(ROOT, args.soft_labels))
    scols = [c for c in df.columns if c.startswith("s_")]
    names = [c[2:] for c in scols]
    S = df[scols].to_numpy(dtype=np.float32)
    dictionary = json.load(open(os.path.join(ROOT, args.dictionary)))
    dictionary_names = [entry["name"] for entry in dictionary]
    if names != dictionary_names:
        raise ValueError("soft-label and dictionary concept order differs")
    flags = {}
    if args.flags:
        flags = json.load(open(os.path.join(ROOT, args.flags))).get("flags", {})
    if len(ids) != len(fi) or len(df) != len(fi):
        raise ValueError("image IDs, embeddings, and labels must have equal row counts")

    img_ad = Adapter(
        fi.shape[1],
        ck["config"]["embed_dim"],
        ck["config"].get("hidden_dim", 1024),
    )
    img_ad.load_state_dict(ck["img_adapter"])
    img_ad.eval()
    with torch.no_grad():
        Z = img_ad(torch.from_numpy(fi)).numpy()

    tr = (df.split == "train").to_numpy()
    gallery_mask = (df.split == args.gallery_split).to_numpy()

    is_reverse = (
        ck["config"].get("objective") == "reverse-ridge"
        or "final_reverse_ridge" in ck
    )
    if is_reverse:
        checkpoint_names = ck.get("concept_names")
        if checkpoint_names is not None and checkpoint_names != names:
            raise ValueError("checkpoint and soft-label concept order differs")
        D = l2(np.load(os.path.join(ROOT, args.run_dir, "concept_directions.npy")))
        if D.shape != (len(names), ck["config"]["embed_dim"]):
            raise ValueError("reverse-ridge direction array has the wrong shape")
        direction_label = "reverse-ridge partial-effect directions"
    else:
        dirs = []
        for k in range(len(names)):
            s = S[tr, k]
            mp = (s[:, None] * Z[tr]).sum(0) / max(s.sum(), 1e-3)
            mn = ((1 - s)[:, None] * Z[tr]).sum(0) / max((1 - s).sum(), 1e-3)
            dirs.append(l2(mp - mn))
        D = np.stack(dirs)
        direction_label = "group-mean directions"
    P_tr = Z[tr] @ D.T
    mu, sd = P_tr.mean(0), P_tr.std(0) + 1e-6

    gal = []
    if args.cc0:
        # public gallery: CC0 subset (redistributable), one image per concept
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
        with torch.no_grad():
            Z_cc0 = img_ad(torch.from_numpy(fi_cc0)).numpy()
        P_gal = (Z_cc0 @ D.T - mu) / sd
        cc0_dir = os.path.join(ROOT, "data/raw/object_images_CC0")
        import re
        for j, fn in enumerate(cc0_ids):
            present = set(by_id[fn]["present"])
            top = [names[k] for k in np.argsort(-P_gal[j]) if names[k] in present][:3]
            concept = re.sub(r"\d+$", "", os.path.splitext(fn)[0]).replace("_", " ")
            gal.append({"img": thumbnail_data_url(os.path.join(cc0_dir, fn), size=110, quality=72),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": concept, "top": top})
    else:
        gallery_idx = np.where(gallery_mask)[0]
        rng = np.random.default_rng(1)
        pick = rng.permutation(gallery_idx)[:args.gallery]
        P_gal = (Z[pick] @ D.T - mu) / sd                # z-scored projections
        for j, i in enumerate(pick):
            top = [names[k] for k in np.argsort(-S[i])[:3]]
            gal.append({"img": thumbnail_data_url(os.path.join(IMG_DIR, ids[i]), size=110, quality=72),
                        "p": [round(float(x), 2) for x in P_gal[j]],
                        "concept": ids[i].split(os.sep)[0].replace("_", " "),
                        "top": top})

    data = {
        "names": names,
        "flags": flags,
        "gallery": gal,
        "meta": {
            "encoder": ck["config"].get("encoder", "unknown"),
            "checkpoint": os.path.basename(args.run_dir.rstrip("/")),
        },
    }
    html = HTML.replace("__DATA__", json.dumps(data)).replace(
        "__DIRECTION_LABEL__", direction_label
    ).replace("__MODEL_LABEL__", "SigLIP2 Giant · selected tuned reverse-ridge checkpoint")
    if args.cc0:
        html = html.replace("<body>", "<body>" + public_nav("playground"))
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
 <h2>Nearest objects (__DIRECTION_LABEL__)</h2>
 <div class="hint">__MODEL_LABEL__. Sliders are nonnegative concept coefficients (0–3).
 Load an item to see the positive part of its projection profile, then edit any axis and retrieve.
 Struck-through sliders = flagged degenerate axes. Green theme = trained adapter.</div>
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
 const inp=document.createElement('input');inp.type='range';inp.min=0;inp.max=3;inp.step=0.1;inp.value=0;
 const val=document.createElement('span');val.className='v';val.textContent='0.0';
 inp.oninput=()=>{val.textContent=(+inp.value).toFixed(1);row.classList.toggle('active',+inp.value!==0);update();};
 row.append(lab,inp,val);box.appendChild(row);sl.push({inp,val,row});
}
function filter(q){q=q.toLowerCase();
 for(const s of sl)s.row.style.display=s.row.dataset.name.includes(q)?'':'none';}
function setSlider(k,v){v=Math.max(0,Math.min(3,v));
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
  c.innerHTML=`<img src="${g.img}"><div class="n">${g.concept}</div>
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
