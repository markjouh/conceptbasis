"""Stage 7 inspection — Render frequencies for raw open-set VLM tags.

This report audits dictionary-discovery input; it does not show the exhaustive
fixed-dictionary label matrix used for training directions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os

from conceptbasis.splits import image_class


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def attribute_statistics(rows: list[dict]) -> list[dict]:
    valid_rows = [row for row in rows if row.get("attributes")]
    class_totals = Counter(image_class(row["image_id"]) for row in valid_rows)
    image_support = Counter()
    class_counts: dict[str, Counter] = defaultdict(Counter)
    for row in valid_rows:
        concept = image_class(row["image_id"])
        for attribute in set(row["attributes"]):
            image_support[attribute] += 1
            class_counts[attribute][concept] += 1

    n_images = len(valid_rows)
    n_classes = len(class_totals)
    statistics = []
    for attribute, support in image_support.items():
        per_class = class_counts[attribute]
        balanced_prevalence = sum(
            count / class_totals[concept]
            for concept, count in per_class.items()
        ) / n_classes
        statistics.append(
            {
                "attribute": attribute,
                "images": support,
                "image_prevalence": support / n_images,
                "classes": len(per_class),
                "class_coverage": len(per_class) / n_classes,
                "balanced_prevalence": balanced_prevalence,
            }
        )
    statistics.sort(
        key=lambda row: (-row["images"], -row["classes"], row["attribute"])
    )
    for rank, row in enumerate(statistics, 1):
        row["rank"] = rank
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--open-tags",
        "--attrs",
        dest="open_tags",
        default="data/attributes_train_vllm_gemma4_nvfp4_open_tags_nonredundant_v8.jsonl",
    )
    parser.add_argument(
        "--out",
        default="outputs/reviews/gemma4-full-train-attribute-frequency.html",
    )
    parser.add_argument("--default-limit", type=int, default=512)
    args = parser.parse_args()
    if args.default_limit < 1:
        parser.error("--default-limit must be positive")

    attrs_path = os.path.join(ROOT, args.open_tags)
    rows = [json.loads(line) for line in open(attrs_path) if line.strip()]
    statistics = attribute_statistics(rows)
    n_classes = len({image_class(row["image_id"]) for row in rows})
    metadata_path = attrs_path + ".meta.json"
    metadata = json.load(open(metadata_path)) if os.path.exists(metadata_path) else {}
    payload = {
        "statistics": statistics,
        "summary": {
            "images": len(rows),
            "classes": n_classes,
            "attributes": len(statistics),
            "defaultLimit": args.default_limit,
            "model": metadata.get("model"),
            "runId": metadata.get("run_id"),
        },
    }
    output = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as file:
        file.write(HTML.replace("__PAYLOAD__", json.dumps(payload)))
    print(f"wrote {output} ({len(statistics)} raw attributes)")


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Raw attribute frequencies</title>
<style>
:root{color-scheme:dark;--bg:#101318;--panel:#171c23;--line:#2b3440;--muted:#91a0b2;--text:#edf3fa;--blue:#71b7ff}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:rgba(16,19,24,.96);border-bottom:1px solid var(--line);padding:22px max(24px,calc((100vw - 1100px)/2)) 16px;backdrop-filter:blur(10px)}
h1{font-size:25px;margin:0 0 7px}.summary{color:var(--muted);margin-bottom:16px}.controls{display:flex;gap:10px;flex-wrap:wrap}
input,select{background:var(--panel);border:1px solid var(--line);border-radius:8px;color:var(--text);padding:9px 11px;font:inherit}
input{min-width:300px;flex:1}.wrap{max-width:1100px;margin:20px auto 50px;padding:0 24px}.note{color:var(--muted);margin:0 0 14px;line-height:1.45}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
th{position:sticky;top:140px;background:#1d242d;color:#aebbc9;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
th:nth-child(2),td:nth-child(2){text-align:left}tr:hover{background:#1b222b}.rank{color:#718095}.tag{font-weight:650;color:#dcebfb}
.meter{height:5px;background:#26303b;border-radius:4px;overflow:hidden;margin-top:5px;width:180px}.meter>i{display:block;height:100%;background:linear-gradient(90deg,#378fe9,#86c7ff)}
.empty{text-align:center;padding:50px;color:var(--muted)}
@media(max-width:750px){.optional{display:none}th{top:170px}.meter{width:100px}input{min-width:180px}}
</style></head><body>
<header><h1>Raw Gemma attribute candidates</h1><div class="summary" id="summary"></div>
<div class="controls"><input id="search" placeholder="Filter raw phrases…" autofocus>
<select id="sort"><option value="images">Sort: image support</option><option value="classes">Sort: class support</option><option value="balanced">Sort: class-balanced prevalence</option><option value="name">Sort: name</option></select>
<select id="limit"><option>128</option><option>256</option><option selected>512</option><option value="99999">All</option></select></div></header>
<main class="wrap"><p class="note">Image support counts tagged exemplars. Class support counts distinct THINGS classes. Class-balanced prevalence gives every class equal total weight, preventing classes with more photographs from dominating.</p>
<table><thead><tr><th>Rank</th><th>Raw phrase</th><th>Images</th><th>Image %</th><th class="optional">Classes</th><th class="optional">Class %</th><th>Balanced %</th></tr></thead><tbody id="rows"></tbody></table></main>
<script>
const P=__PAYLOAD__, D=P.statistics, S=P.summary;
const q=document.getElementById('search'), sort=document.getElementById('sort'), limit=document.getElementById('limit'), body=document.getElementById('rows');
limit.value=String(S.defaultLimit);
document.getElementById('summary').textContent=`${S.attributes.toLocaleString()} distinct phrases · ${S.images.toLocaleString()} valid train images · ${S.classes.toLocaleString()} train classes · ${S.model||'unknown model'}`;
const pct=x=>(100*x).toFixed(2)+'%';
function render(){
 let rows=D.filter(x=>x.attribute.includes(q.value.trim().toLowerCase()));
 const key=sort.value;
 rows.sort(key==='name'?(a,b)=>a.attribute.localeCompare(b.attribute):key==='classes'?(a,b)=>b.classes-a.classes||b.images-a.images:key==='balanced'?(a,b)=>b.balanced_prevalence-a.balanced_prevalence||b.images-a.images:(a,b)=>a.rank-b.rank);
 rows=rows.slice(0,Number(limit.value)); body.innerHTML='';
 if(!rows.length){const td=document.createElement('td');td.colSpan=7;td.className='empty';td.textContent='No matching phrases';const tr=document.createElement('tr');tr.appendChild(td);body.appendChild(tr);return}
 for(const x of rows){const tr=document.createElement('tr');
  const values=[x.rank,x.attribute,x.images.toLocaleString(),pct(x.image_prevalence),x.classes.toLocaleString(),pct(x.class_coverage),pct(x.balanced_prevalence)];
  values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===0)td.className='rank';if(i===1){td.className='tag';const m=document.createElement('div');m.className='meter';const b=document.createElement('i');b.style.width=(100*x.images/D[0].images)+'%';m.appendChild(b);td.appendChild(m)}if(i===4||i===5)td.classList.add('optional');tr.appendChild(td)});body.appendChild(tr)}
}
q.addEventListener('input',render);sort.addEventListener('change',render);limit.addEventListener('change',render);render();
</script></body></html>'''


if __name__ == "__main__":
    main()
