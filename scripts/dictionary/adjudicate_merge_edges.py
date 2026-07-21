"""Optional Stage 2 experiment — Judge proposed phrase merges with a local LLM.

Consumes the complete proposal artifact from ``propose_merge_edges.py`` and
writes resumable, phrase-bound decisions for the dictionary builder's optional
``adjudicated`` merge method. It is not part of the accepted profile-only run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conceptbasis.encoders import sha256_file, write_json_atomic
from conceptbasis.vlm import (
    GEMMA_MODEL_ID,
    LOCAL_VLLM_API_URL,
    check_or_write_metadata,
    session,
    sha256_text,
)
from scripts.dictionary.build_dictionary import ROOT


STRICT_EQUIVALENCE_SYSTEM = """You are a meticulous ontology editor for a visual object-property dictionary.
Judge whether each pair should be ONE shared semantic axis. Approve only when the labels are
interchangeable for this purpose: spelling/inflection variants, material adjective/noun forms,
or genuine synonyms and near-synonyms whose distinction would not be useful as separate visual
properties. Reject merely related or co-occurring properties; broader/narrower categories;
different materials, colors, degrees, functions, or object types; causes/effects; and opposites.
Examples: gray/grey=true, wood/wooden=true, round/circular=true, small/tiny=true,
rigid/lightweight=false, brass/bronze=false, transparent/translucent=false,
household appliance/household item=false, paper/paperwork=false.
Return exactly one JSON object mapping every supplied edge ID to true or false. No prose."""

COHERENT_AXIS_SYSTEM = """You are reviewing proposed edges for a visual object-property dictionary.
This is NOT strict symbolic synonym matching. Approve an edge when the two labels are genuine
synonyms OR are so closely related perceptually or practically that pooling their image examples
would form one coherent, useful axis. Closely related shades, materials, shapes, intensity variants,
subtypes, and broad/narrow functional labels may be approved. Reject only unrelated collisions,
tenuous associations, and independent dimensions that merely co-occur on the same objects.
Examples: gray/grey=true, wood/wooden=true, round/circular=true, small/tiny=true,
brass/bronze=true, transparent/translucent=true, animal/mammal=true,
rigid/lightweight=false, casual/perennial=false, containment/sweet=false,
hard/open=false, monochrome/organic=false.
Return exactly one JSON object mapping every supplied edge ID to true or false. No prose."""

POSITIVE_FAMILY_SYSTEM = """You are filtering candidate edges for a positive-only visual concept
dictionary. Approve when both labels belong in ONE useful positive concept family: true synonyms,
spelling/inflection variants, material noun/adjective variants, light/dark or intensity variants
of the same underlying color/property, or extremely close neighboring colors/materials whose
separation adds little value. Reject labels that merely co-occur, lie on independent dimensions,
are opposites, or are only thematically/taxonomically related. Examples true: brown/light brown,
yellow/yellowish, blue/light blue, brass/bronze, gray/grey, wood/wooden,
transparent/translucent. Examples false: rigid/heavy, smooth/soft, natural/perishable,
long/thin, food/plant, tool/machine, container/kitchen tool, metallic/shiny,
monochrome/organic. Return exactly one JSON object mapping every supplied edge ID to true or
false. No prose."""

POLICIES = {
    "strict-equivalence": STRICT_EQUIVALENCE_SYSTEM,
    "coherent-axis": COHERENT_AXIS_SYSTEM,
    "positive-family": POSITIVE_FAMILY_SYSTEM,
}


def parse_decisions(content: str, expected_ids: list[str]) -> dict[str, bool]:
    text = re.sub(r"```(?:json)?|```", "", content).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict) or set(value) != set(expected_ids):
        raise ValueError("response edge IDs do not exactly match the batch")
    if any(type(decision) is not bool for decision in value.values()):
        raise ValueError("all edge decisions must be booleans")
    return value


def judge_batch(
    edges: list[dict], *, api_url: str, model: str, api_key: str, retries: int,
    system: str,
) -> dict[str, bool]:
    ids = [edge["edge_id"] for edge in edges]
    compact = [[edge["edge_id"], edge["left"], edge["right"]] for edge in edges]
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max(512, len(edges) * 20),
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(compact, separators=(",", ":"))},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session().post(api_url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            return parse_decisions(response.json()["choices"][0]["message"]["content"], ids)
        except Exception as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    if len(edges) > 1:
        midpoint = len(edges) // 2
        return {
            **judge_batch(
                edges[:midpoint], api_url=api_url, model=model, api_key=api_key,
                retries=retries, system=system,
            ),
            **judge_batch(
                edges[midpoint:], api_url=api_url, model=model, api_key=api_key,
                retries=retries, system=system,
            ),
        }
    raise RuntimeError(f"failed to adjudicate edge after {retries} attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--api-url", default=LOCAL_VLLM_API_URL)
    parser.add_argument("--model", default=GEMMA_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--policy", choices=tuple(POLICIES), default="coherent-axis")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.workers, args.retries) < 1:
        parser.error("batch size, workers, and retries must be positive")

    proposals_path = ROOT / args.proposals
    proposals = json.loads(proposals_path.read_text())
    if proposals.get("schema") != "conceptbasis.merge-proposals/v1":
        raise ValueError("unsupported merge proposal schema")
    edges = proposals["edges"]
    system = POLICIES[args.policy]
    if proposals.get("n_proposed_edges") != len(edges):
        raise ValueError("merge proposal artifact is incomplete")

    output = ROOT / args.out
    partial = output.with_suffix(output.suffix + ".partial.jsonl")
    metadata = output.with_suffix(output.suffix + ".meta.json")
    expected_meta = {
        "schema": "conceptbasis.merge-adjudication-run/v1",
        "proposals": args.proposals,
        "proposals_sha256": sha256_file(proposals_path),
        "model": args.model,
        "model_revision": args.model_revision,
        "api_url": args.api_url,
        "policy": args.policy,
        "prompt_sha256": sha256_text(system),
        "batch_size": args.batch_size,
    }
    check_or_write_metadata(metadata, expected_meta, output_path=partial)
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["edge_id"]] = row
    todo = [edge for edge in edges if edge["edge_id"] not in done]
    batches = [todo[start:start + args.batch_size] for start in range(0, len(todo), args.batch_size)]
    print(f"{len(edges)} proposals: {len(done)} done, {len(todo)} to adjudicate")

    api_key = os.environ.get("VLM_API_KEY") or "local"
    lock = threading.Lock()
    started = time.monotonic()
    with partial.open("a") as file, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                judge_batch,
                batch,
                api_url=args.api_url,
                model=args.model,
                api_key=api_key,
                retries=args.retries,
                system=system,
            ): batch
            for batch in batches
        }
        for future in tqdm(as_completed(futures), total=len(futures), unit="batch"):
            batch = futures[future]
            decisions = future.result()
            rows = [
                {
                    "edge_id": edge["edge_id"],
                    "equivalent": decisions[edge["edge_id"]],
                }
                for edge in batch
            ]
            with lock:
                for row in rows:
                    file.write(json.dumps(row, sort_keys=True) + "\n")
                    done[row["edge_id"]] = row
                file.flush()

    ordered = []
    for edge in edges:
        decision = done.get(edge["edge_id"])
        if decision is None:
            raise ValueError(f"missing decision for {edge['edge_id']}")
        ordered.append({**edge, "equivalent": decision["equivalent"]})
    accepted = sum(row["equivalent"] for row in ordered)
    write_json_atomic(
        output,
        {
            "schema": "conceptbasis.merge-adjudication/v1",
            "model": args.model,
            "model_revision": args.model_revision,
            "policy": args.policy,
            "prompt": system,
            "prompt_sha256": sha256_text(system),
            "proposals": args.proposals,
            "proposals_sha256": sha256_file(proposals_path),
            "phrases": proposals["phrases"],
            "text_cosine_threshold": proposals["text_cosine_threshold"],
            "n_proposed_edges": len(ordered),
            "n_accepted_edges": accepted,
            "decisions": ordered,
        },
    )
    elapsed = time.monotonic() - started
    print(
        f"wrote {args.out}: accepted {accepted}/{len(ordered)} edges "
        f"in {elapsed:.1f}s ({len(ordered) / max(elapsed, 1e-9):.1f} edges/s)"
    )


if __name__ == "__main__":
    main()
