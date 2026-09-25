"""Domain-balanced batching, shared by Task 2 and Task 3.

Task 2: each adaptation batch = 8 examples from each of the 3 labeled sources
+ 24 unlabeled target examples (equal 24/24 source/target total), cycling
whichever domain's loader is shorter.
Task 3: each batch = 8 examples from each of the 3 sources (no target).
"""
from __future__ import annotations

from typing import Iterator, Sequence

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    DataLoader = object
    Dataset = object


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


class DomainBalancedBatchIterator:
    """Wraps one DataLoader per domain (each already shuffling with its own
    per-domain batch size) and yields a dict of per-domain batches per step,
    cycling shorter loaders so every step has a full complement from every
    domain. Iterate for `steps_per_epoch` steps, then re-create for the next
    epoch (or just keep pulling from the same infinite generator).
    """

    def __init__(self, domain_loaders: dict[str, "DataLoader"], steps_per_epoch: int):
        self.domain_loaders = domain_loaders
        self.steps_per_epoch = steps_per_epoch
        self._iters = {name: _cycle(loader) for name, loader in domain_loaders.items()}

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self.steps_per_epoch):
            yield {name: next(it) for name, it in self._iters.items()}


def build_uda_batch_iterator(source_datasets: dict[str, "Dataset"], target_dataset: "Dataset",
                              per_source_batch: int = 8, target_batch: int = 24,
                              seed: int = 6304, num_workers: int = 0):
    """Task 2 protocol: {Photo, ArtPainting, Cartoon} @ per_source_batch=8 each,
    plus target (Sketch, unlabeled) @ target_batch=24. steps_per_epoch is set by
    the *largest* source domain so every source example is seen at least once
    per epoch (smaller domains and the target cycle to keep up)."""
    from common.seed import seeded_generator

    loaders = {}
    max_len = 0
    for name, ds in source_datasets.items():
        loaders[name] = DataLoader(ds, batch_size=per_source_batch, shuffle=True,
                                    num_workers=num_workers, drop_last=True,
                                    generator=seeded_generator(seed))
        max_len = max(max_len, len(loaders[name]))
    loaders["target"] = DataLoader(target_dataset, batch_size=target_batch, shuffle=True,
                                    num_workers=num_workers, drop_last=True,
                                    generator=seeded_generator(seed))
    steps_per_epoch = max(max_len, 1)
    return DomainBalancedBatchIterator(loaders, steps_per_epoch)


def build_dg_batch_iterator(source_datasets: dict[str, "Dataset"], per_source_batch: int = 8,
                             seed: int = 6304, num_workers: int = 0):
    """Task 3 protocol: {Photo, ArtPainting, Cartoon} @ 8 each, no target."""
    from common.seed import seeded_generator

    loaders = {}
    max_len = 0
    for name, ds in source_datasets.items():
        loaders[name] = DataLoader(ds, batch_size=per_source_batch, shuffle=True,
                                    num_workers=num_workers, drop_last=True,
                                    generator=seeded_generator(seed))
        max_len = max(max_len, len(loaders[name]))
    steps_per_epoch = max(max_len, 1)
    return DomainBalancedBatchIterator(loaders, steps_per_epoch)


def concat_domain_batches(batch_dict: dict[str, tuple], device: str = "cpu"):
    """Concatenates a dict of (images, labels) per-domain batches from
    DomainBalancedBatchIterator into flat (images, labels, domain_ids) tensors,
    domain_ids indexing into sorted(batch_dict.keys()) -- convenient for
    methods (DAN/DANN/CDAN) that need to know which rows came from which
    domain within a single forward pass."""
    names = sorted(batch_dict.keys())
    all_images, all_labels, all_domain_ids = [], [], []
    for domain_id, name in enumerate(names):
        images, labels = batch_dict[name]
        all_images.append(images)
        all_labels.append(labels)
        all_domain_ids.append(torch.full((images.shape[0],), domain_id, dtype=torch.long))
    images = torch.cat(all_images, dim=0).to(device)
    labels = torch.cat(all_labels, dim=0).to(device)
    domain_ids = torch.cat(all_domain_ids, dim=0).to(device)
    return images, labels, domain_ids, names
