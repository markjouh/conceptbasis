"""Stage 7 inspection — Render exhaustive fixed-dictionary TRUE/FALSE labels.

The page deliberately consumes the closed-set ``present`` decisions rather
than the open tags used to construct the dictionary. It provides a
literal image-by-concept matrix plus a complete 256-way TRUE/FALSE view for
each held-out object.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from conceptbasis.site import public_nav, thumbnail_data_url
from conceptbasis.splits import load_split_manifest, split_for_image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DICTIONARY = "data/dictionary_usage_profile_v8.json"
DEFAULT_LABELS = (
    "data/dictionary_labels_cc0_dev_vllm_gemma4_nvfp4_usage_profile_v8_"
    "object_grounded_v11.jsonl"
)
def validate_fixed_rows(
    dictionary: list[dict],
    rows: list[dict],
    expected_ids: list[str],
) -> tuple[list[str], list[dict]]:
    """Validate and order a complete binary label matrix.

    The returned rows follow ``expected_ids`` so matrix order is stable across
    regeneration.  UNKNOWN/UNCERTAIN values are rejected because this viewer
    promises a final boolean matrix.
    """
    names = [entry["name"] for entry in dictionary]
    if not names or len(names) != len(set(names)):
        raise ValueError("dictionary names must be non-empty and unique")
    name_set = set(names)
    by_id: dict[str, dict] = {}
    for row in rows:
        image_id = row.get("image_id")
        if not image_id or image_id in by_id:
            raise ValueError(f"missing or duplicate image_id: {image_id!r}")
        if row.get("status") != "ok":
            raise ValueError(f"non-final row for {image_id}: {row.get('status')!r}")
        present = row.get("present")
        uncertain = row.get("uncertain", [])
        if not isinstance(present, list) or not isinstance(uncertain, list):
            raise ValueError(f"labels must be lists for {image_id}")
        if uncertain:
            raise ValueError(f"uncertain labels prevent a boolean matrix: {image_id}")
        if len(present) != len(set(present)):
            raise ValueError(f"duplicate positive label for {image_id}")
        unknown = set(present) - name_set
        if unknown or row.get("unknown_names"):
            raise ValueError(f"unknown labels for {image_id}: {sorted(unknown)}")
        by_id[image_id] = row

    expected_set = set(expected_ids)
    if len(expected_ids) != len(expected_set):
        raise ValueError("expected image IDs must be unique")
    missing = expected_set - set(by_id)
    extra = set(by_id) - expected_set
    if missing or extra:
        raise ValueError(
            f"fixed-label coverage mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    return names, [by_id[image_id] for image_id in expected_ids]


def build_payload(
    names: list[str], rows: list[dict], image_dir: str
) -> tuple[list[dict], dict[str, float]]:
    items = []
    positive_counts = []
    for row in rows:
        image_id = row["image_id"]
        path = os.path.join(image_dir, image_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        present = set(row["present"])
        bits = "".join("1" if name in present else "0" for name in names)
        positive_counts.append(len(present))
        items.append(
            {
                "id": image_id,
                "label": row.get("object_label") or Path(image_id).stem.replace("_", " "),
                "img": thumbnail_data_url(path, size=256),
                "bits": bits,
                "yes": len(present),
            }
        )
    stats = {
        "mean": sum(positive_counts) / len(positive_counts),
        "min": min(positive_counts),
        "max": max(positive_counts),
    }
    return items, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY)
    parser.add_argument(
        "--fixed-labels", "--labels", dest="fixed_labels", default=DEFAULT_LABELS
    )
    parser.add_argument("--image-ids", default="data/cc0_image_ids.json")
    parser.add_argument("--image-dir", default="data/raw/object_images_CC0")
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--out", default="docs/fixed-labels.html")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("rendering test requires --allow-test")

    def absolute(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(ROOT, path)

    with open(absolute(args.dictionary)) as file:
        dictionary = json.load(file)
    with open(absolute(args.fixed_labels)) as file:
        rows = [json.loads(line) for line in file if line.strip()]
    with open(absolute(args.image_ids)) as file:
        all_image_ids = json.load(file)
    manifest = load_split_manifest(ROOT, args.split_manifest)
    expected_ids = [
        image_id
        for image_id in all_image_ids
        if split_for_image(manifest, image_id) == args.split
    ]
    names, rows = validate_fixed_rows(dictionary, rows, expected_ids)
    items, stats = build_payload(names, rows, absolute(args.image_dir))
    payload = {
        "names": names,
        "items": items,
        "split": args.split,
        "source": os.path.basename(args.fixed_labels),
        "stats": stats,
    }
    output = absolute(args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as file:
        file.write(HTML.replace("__NAV__", public_nav("fixed-labels")).replace("__DATA__", json.dumps(payload)))
    print(
        f"wrote {output} ({len(items)} × {len(names)} matrix, "
        f"{stats['mean']:.2f} TRUE/image, {os.path.getsize(output) / 1e6:.1f} MB)"
    )


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixed-dictionary labels · Concept Basis</title>
<style>
:root{color-scheme:dark;--bg:#11161b;--panel:#181f26;--line:#2a3540;--text:#e8edf2;--muted:#8e9daa;--yes:#36d487;--yes-bg:#173d2d;--no:#84909a;--no-bg:#202831;--accent:#7fc7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}button,input,select{font:inherit}
#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.96);border:1px solid #38404c;border-radius:20px;padding:6px 14px;display:flex;gap:14px;font-size:12px}
#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover,#sitenav a.here{color:#fff}#sitenav a.here{font-weight:650}
header{padding:54px 28px 20px;max-width:1500px;margin:auto}h1{font-size:27px;margin:0 0 7px}header p{color:var(--muted);margin:0;max-width:920px}.stats{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.stat{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:6px 11px;font-size:12px}.stat b{color:var(--accent)}
.tabs{display:flex;gap:8px;max-width:1500px;margin:0 auto 14px;padding:0 28px}.tabs button,.controls button{border:1px solid var(--line);background:var(--panel);color:var(--muted);border-radius:8px;padding:8px 12px;cursor:pointer}.tabs button.active,.controls button.active{background:#15344a;color:#d8efff;border-color:#326a8f}
main{max-width:1500px;margin:auto;padding:0 28px 40px}.view{display:none}.view.active{display:block}
.detail{display:grid;grid-template-columns:280px minmax(0,1fr);gap:16px}.objects,.inspect,.matrix-wrap{background:var(--panel);border:1px solid var(--line);border-radius:12px}.objects{padding:12px;height:calc(100vh - 205px);min-height:520px;display:flex;flex-direction:column}.objects input,.controls input{width:100%;background:#11171d;border:1px solid var(--line);color:var(--text);border-radius:7px;padding:8px 10px}.object-list{overflow:auto;margin-top:9px}.object-row{display:flex;justify-content:space-between;gap:8px;padding:7px 8px;border-radius:6px;cursor:pointer;text-transform:capitalize}.object-row:hover{background:#222c35}.object-row.active{background:#193a50;color:#d9f0ff}.object-row span:last-child{color:var(--muted);font-variant-numeric:tabular-nums}
.inspect{padding:16px;min-width:0}.hero{display:grid;grid-template-columns:230px 1fr;gap:18px;align-items:start}.hero img{width:230px;height:230px;object-fit:contain;background:#fff;border-radius:9px}.hero h2{font-size:22px;text-transform:capitalize;margin:4px 0}.meta{color:var(--muted)}.legend{display:flex;gap:15px;margin-top:16px;font-size:12px}.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}.dot.y{background:var(--yes)}.dot.n{background:#626f7b}
.controls{display:flex;gap:8px;align-items:center;margin:16px 0 10px}.controls input{max-width:330px;margin-left:auto}.concept-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:6px}.concept{border:1px solid var(--line);border-radius:7px;padding:8px 9px;display:flex;align-items:center;justify-content:space-between;gap:5px;min-width:0}.concept .name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.concept .value{font-size:10px;font-weight:800;letter-spacing:.04em}.concept.yes{background:var(--yes-bg);border-color:#267f58}.concept.yes .value{color:#7ff0b5}.concept.no{background:var(--no-bg);color:#aab3bb}.concept.no .value{color:#778590}
.matrix-wrap{padding:14px}.matrix-note{color:var(--muted);margin:0 0 10px}.matrix-scroller{max-height:calc(100vh - 285px);overflow:auto;border:1px solid var(--line);background:#0e1317}.matrix-stage{position:relative;width:max-content;min-width:100%}canvas{display:block;image-rendering:pixelated}.hover{min-height:22px;color:#c4d2de;margin-top:9px;font-variant-numeric:tabular-nums}.matrix-search{display:flex;gap:8px;margin-bottom:10px}.matrix-search input{background:#11171d;border:1px solid var(--line);color:var(--text);border-radius:7px;padding:8px 10px;max-width:320px;width:100%}
@media(max-width:850px){#sitenav{position:static;margin:8px;overflow:auto;white-space:nowrap}header{padding-top:20px}.detail{grid-template-columns:1fr}.objects{height:260px;min-height:0}.hero{grid-template-columns:140px 1fr}.hero img{width:140px;height:140px}}
</style></head><body>__NAV__
<header><h1>Final fixed-dictionary TRUE/FALSE labels</h1><p>Exhaustive Gemma judgments over the final 256-concept dictionary for every held-out development object. These are the closed-set labels used by evaluation—not the open-set tags used while constructing the dictionary.</p><div class="stats" id="stats"></div></header>
<div class="tabs"><button class="active" data-view="detailView">Per object</button><button data-view="matrixView">Full matrix</button></div>
<main>
<section id="detailView" class="view active"><div class="detail"><aside class="objects"><input id="objectSearch" placeholder="Find an object…"><div id="objectList" class="object-list"></div></aside><article class="inspect"><div class="hero"><img id="heroImage"><div><h2 id="heroLabel"></h2><div id="heroMeta" class="meta"></div><div class="legend"><span><i class="dot y"></i>TRUE</span><span><i class="dot n"></i>FALSE</span></div></div></div><div class="controls"><button class="active" data-filter="all">All 256</button><button data-filter="yes">TRUE</button><button data-filter="no">FALSE</button><input id="conceptSearch" placeholder="Find a concept…"></div><div id="conceptGrid" class="concept-grid"></div></article></div></section>
<section id="matrixView" class="view"><div class="matrix-wrap"><p class="matrix-note">Rows are held-out objects; columns are dictionary concepts in canonical order. Green = TRUE, charcoal = FALSE. Hover for the exact cell; click a row to inspect all 256 decisions.</p><div class="matrix-search"><input id="matrixObjectSearch" placeholder="Highlight object…"><input id="matrixConceptSearch" placeholder="Highlight concept…"></div><div class="matrix-scroller"><div class="matrix-stage"><canvas id="matrix"></canvas></div></div><div id="matrixHover" class="hover">Hover over a cell.</div></div></section>
</main><script>
const DATA=__DATA__, names=DATA.names, items=DATA.items;let selected=0,filter='all';
const stats=document.getElementById('stats');stats.innerHTML=`<span class="stat"><b>${items.length}</b> ${DATA.split} objects</span><span class="stat"><b>${names.length}</b> concepts</span><span class="stat"><b>${DATA.stats.mean.toFixed(2)}</b> TRUE / image</span><span class="stat">range <b>${DATA.stats.min}–${DATA.stats.max}</b></span>`;
function renderObjects(q=''){q=q.toLowerCase();const box=document.getElementById('objectList');box.innerHTML='';items.forEach((it,i)=>{if(!(it.label+' '+it.id).toLowerCase().includes(q))return;const row=document.createElement('div');row.className='object-row'+(i===selected?' active':'');row.innerHTML=`<span>${it.label}</span><span>${it.yes}</span>`;row.onclick=()=>{selected=i;renderObjects(document.getElementById('objectSearch').value);renderDetail()};box.appendChild(row)})}
function renderDetail(){const it=items[selected];document.getElementById('heroImage').src=it.img;document.getElementById('heroLabel').textContent=it.label;document.getElementById('heroMeta').textContent=`${it.id} · ${it.yes} TRUE · ${names.length-it.yes} FALSE`;const q=document.getElementById('conceptSearch').value.toLowerCase();const grid=document.getElementById('conceptGrid');grid.innerHTML='';names.forEach((name,k)=>{const yes=it.bits[k]==='1';if(filter!=='all'&&filter!==(yes?'yes':'no')||!name.includes(q))return;const cell=document.createElement('div');cell.className='concept '+(yes?'yes':'no');cell.innerHTML=`<span class="name" title="${name}">${name}</span><span class="value">${yes?'TRUE':'FALSE'}</span>`;grid.appendChild(cell)})}
document.getElementById('objectSearch').oninput=e=>renderObjects(e.target.value);document.getElementById('conceptSearch').oninput=renderDetail;document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));renderDetail()});
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===b.dataset.view));if(b.dataset.view==='matrixView')drawMatrix()});
const canvas=document.getElementById('matrix'),ctx=canvas.getContext('2d'),cellW=5,cellH=5;canvas.width=names.length*cellW;canvas.height=items.length*cellH;canvas.style.width=canvas.width+'px';canvas.style.height=canvas.height+'px';
function drawMatrix(){ctx.fillStyle='#202831';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#35cf83';items.forEach((it,r)=>{for(let c=0;c<names.length;c++)if(it.bits[c]==='1')ctx.fillRect(c*cellW,r*cellH,cellW,cellH)});const oq=document.getElementById('matrixObjectSearch').value.toLowerCase(),cq=document.getElementById('matrixConceptSearch').value.toLowerCase();if(oq){items.forEach((it,r)=>{if((it.label+' '+it.id).toLowerCase().includes(oq)){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(1,r*cellH+1,canvas.width-2,cellH-2)}})}if(cq){names.forEach((n,c)=>{if(n.includes(cq)){ctx.strokeStyle='#7fc7ff';ctx.lineWidth=2;ctx.strokeRect(c*cellW+1,1,cellW-2,canvas.height-2)}})}}
document.getElementById('matrixObjectSearch').oninput=drawMatrix;document.getElementById('matrixConceptSearch').oninput=drawMatrix;canvas.onmousemove=e=>{const rect=canvas.getBoundingClientRect(),c=Math.floor((e.clientX-rect.left)*canvas.width/rect.width/cellW),r=Math.floor((e.clientY-rect.top)*canvas.height/rect.height/cellH);if(items[r]&&names[c])document.getElementById('matrixHover').textContent=`${items[r].label} × ${names[c]} → ${items[r].bits[c]==='1'?'TRUE':'FALSE'} · row ${r+1}, column ${c+1}`};canvas.onclick=e=>{const rect=canvas.getBoundingClientRect(),r=Math.floor((e.clientY-rect.top)*canvas.height/rect.height/cellH);if(items[r]){selected=r;renderObjects();renderDetail();document.querySelector('[data-view="detailView"]').click()}};
renderObjects();renderDetail();
</script></body></html>"""


if __name__ == "__main__":
    main()
