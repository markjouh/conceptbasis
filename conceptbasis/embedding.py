"""Shared image preprocessing primitives for frozen-encoder jobs."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
from PIL import Image


PathLike = str | Path


def load_image(path: PathLike, preprocess: Callable) -> torch.Tensor:
    """Open an image safely and apply an encoder's preprocessing transform."""
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def image_batches(
    paths: Sequence[PathLike],
    preprocess: Callable,
    batch_size: int,
    workers: int,
    prefetch_batches: int,
) -> Iterator[torch.Tensor]:
    """Yield ordered image batches with bounded, optional CPU prefetching."""
    if batch_size < 1 or workers < 0 or prefetch_batches < 1:
        raise ValueError("batch size/prefetch must be positive and workers nonnegative")
    batches = [
        paths[start : start + batch_size]
        for start in range(0, len(paths), batch_size)
    ]
    if workers == 0:
        for batch_paths in batches:
            yield torch.stack([load_image(path, preprocess) for path in batch_paths])
        return

    def submit_batch(
        executor: ThreadPoolExecutor,
        batch_paths: Sequence[PathLike],
    ) -> list[Future[torch.Tensor]]:
        return [executor.submit(load_image, path, preprocess) for path in batch_paths]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[list[Future[torch.Tensor]]] = deque()
        next_batch = 0
        while next_batch < min(prefetch_batches, len(batches)):
            pending.append(submit_batch(executor, batches[next_batch]))
            next_batch += 1
        while pending:
            futures = pending.popleft()
            if next_batch < len(batches):
                pending.append(submit_batch(executor, batches[next_batch]))
                next_batch += 1
            yield torch.stack([future.result() for future in futures])
