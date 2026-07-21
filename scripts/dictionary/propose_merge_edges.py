"""Optional Stage 2 experiment — Propose phrase pairs for merge adjudication.

This is not part of the accepted usage-profile recipe. It retrieves liberal
text-cosine neighbors so ``adjudicate_merge_edges.py`` can test an LLM-filtered
merge baseline without changing dictionary selection or model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conceptbasis.encoders import select_encoder, sha256_file, write_json_atomic
from conceptbasis.train import require_cuda
from scripts.dictionary.build_dictionary import (
    ROOT,
    candidate_phrases,
    cosine_similarity_matrix,
    encode_phrases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--open-tags",
        "--attrs",
        dest="open_tags",
        default="data/attributes_train_vllm_gemma4_nvfp4_open_tags_nonredundant_v8.jsonl",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", default="siglip2-giant")
    parser.add_argument("--min-mentions", type=int, default=3)
    parser.add_argument("--min-class-support", type=int, default=3)
    parser.add_argument("--text-cosine", type=float, default=0.88)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not -1 <= args.text_cosine <= 1:
        parser.error("--text-cosine must be between -1 and 1")
    require_cuda()

    attrs_path = ROOT / args.open_tags
    rows = [
        json.loads(line)
        for line in attrs_path.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("attributes")]
    phrases, negated_of, _support = candidate_phrases(
        rows,
        min_mentions=args.min_mentions,
        min_class_support=args.min_class_support,
    )
    encoder = select_encoder(args.encoder)
    embeddings, source = encode_phrases(
        phrases,
        encoder,
        device="cuda",
        precision=args.precision,
        batch_size=args.batch_size,
    )
    similarity = cosine_similarity_matrix(embeddings)
    left, right = np.triu_indices(len(phrases), 1)
    selected = np.flatnonzero(similarity[left, right] >= args.text_cosine)
    ranked = sorted(
        selected,
        key=lambda index: (
            -float(similarity[left[index], right[index]]),
            phrases[left[index]],
            phrases[right[index]],
        ),
    )
    edges = [
        {
            "edge_id": f"e{position:06d}",
            "left": phrases[left[index]],
            "right": phrases[right[index]],
            "text_cosine": float(similarity[left[index], right[index]]),
        }
        for position, index in enumerate(ranked)
    ]
    output = ROOT / args.out
    write_json_atomic(
        output,
        {
            "schema": "conceptbasis.merge-proposals/v1",
            "encoder": encoder.as_dict(),
            "encoder_source": source,
            "open_tags": args.open_tags,
            "attrs_sha256": sha256_file(attrs_path),
            "min_mentions": args.min_mentions,
            "min_class_support": args.min_class_support,
            "text_cosine_threshold": args.text_cosine,
            "negation_policy": "exclude",
            "excluded_lexical_negations": negated_of,
            "phrases": phrases,
            "n_proposed_edges": len(edges),
            "edges": edges,
        },
    )
    print(f"wrote {args.out}: {len(phrases)} phrases, {len(edges)} proposed edges")


if __name__ == "__main__":
    main()
