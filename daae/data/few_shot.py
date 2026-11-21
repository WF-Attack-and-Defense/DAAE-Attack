"""Few-shot episodic sampling for meta-learning DAAE training."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from daae.config import DAAEConfig
from daae.data.cache import StackedPairedDataset


def build_label_index(labels: np.ndarray) -> Dict[int, List[int]]:
    index: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels.tolist()):
        index[int(label)].append(i)
    return dict(index)


class FewShotWrapper(Dataset):
    """
    Wraps a stacked dataset and adds K support examples from the same class.

    Each sample is a query trace; support traces teach the model how this
    defense perturbs traffic for that website (few-shot meta-learning).

    When `support_base` differs from `base` (e.g. val queries with train
    supports), supports are drawn only from `support_base`.
    """

    def __init__(
        self,
        base: StackedPairedDataset,
        label_index: Dict[int, List[int]],
        support_shots: int = 5,
        seed: int = 42,
        eval_mode: bool = False,
        support_base: Optional[StackedPairedDataset] = None,
    ):
        self.base = base
        self.support_base = support_base or base
        self.label_index = label_index
        self.support_shots = support_shots
        self.eval_mode = eval_mode
        self.rng = np.random.default_rng(seed)
        self._all_ids = [i for ids in label_index.values() for i in ids]
        self._id_label = {
            i: int(lab) for lab, ids in label_index.items() for i in ids
        }

    def __len__(self) -> int:
        return len(self.base)

    def _sample_same_class(self, label: int, exclude: Optional[int]) -> List[int]:
        pool = list(self.label_index.get(label, []))
        if self.support_base is self.base and exclude is not None:
            pool = [i for i in pool if i != exclude]
        if not pool:
            pool = list(self.label_index.get(label, [0]))
        k = min(self.support_shots, len(pool))
        if self.eval_mode:
            return sorted(pool)[:k]
        if k <= 0:
            return []
        return self.rng.choice(pool, size=k, replace=False).tolist()

    def _sample_mixed_class(self, label: int, exclude: Optional[int]) -> List[int]:
        """Defense dummy pattern from other websites — query class is unused."""
        pool = [
            i
            for i in self._all_ids
            if self._id_label.get(i) != label and i != exclude
        ]
        if len(pool) < self.support_shots:
            pool = [i for i in self._all_ids if i != exclude] or list(self._all_ids)
        k = min(self.support_shots, len(pool))
        if k <= 0:
            return []
        if self.eval_mode:
            step = max(1, len(pool) // k)
            return [pool[(j * step) % len(pool)] for j in range(k)]
        return self.rng.choice(pool, size=k, replace=False).tolist()

    def _sample_support(self, label: int, exclude: Optional[int] = None) -> List[int]:
        # Dummy injection is defense-specific. Prefer mixed-class supports so
        # the mask head does not need the query website id at recover time.
        if self.eval_mode:
            return self._sample_mixed_class(label, exclude)
        if float(self.rng.random()) < 0.25:
            return self._sample_same_class(label, exclude)
        return self._sample_mixed_class(label, exclude)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query = self.base[idx]
        label = int(query["label"].item())
        exclude = idx if self.support_base is self.base else None
        support_ids = self._sample_support(label, exclude)
        if not support_ids:
            support_ids = self._sample_mixed_class(label, exclude) or [0]

        support_defended = torch.stack([self.support_base[i]["defended"] for i in support_ids])
        support_mask = torch.stack([self.support_base[i]["dummy_mask"] for i in support_ids])

        return {
            **query,
            "support_defended": support_defended,
            "support_mask": support_mask,
        }


def collate_few_shot(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for key in keys:
        if key in ("support_defended", "support_mask"):
            out[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def wrap_few_shot(
    dataset: Dataset,
    config: DAAEConfig,
    eval_mode: bool = False,
    support_dataset: Optional[StackedPairedDataset] = None,
) -> Dataset:
    if not isinstance(dataset, StackedPairedDataset):
        return dataset
    support_source = support_dataset or dataset
    labels = support_source.labels.cpu().numpy()
    label_index = build_label_index(labels)
    return FewShotWrapper(
        dataset,
        label_index,
        support_shots=config.support_shots,
        seed=config.seed,
        eval_mode=eval_mode,
        support_base=support_source,
    )


def build_fs_dataloader(
    dataset: Dataset,
    config: DAAEConfig,
    shuffle: bool,
    eval_mode: bool = False,
    support_dataset: Optional[StackedPairedDataset] = None,
) -> DataLoader:
    dataset = wrap_few_shot(
        dataset, config, eval_mode=eval_mode, support_dataset=support_dataset
    )
    on_gpu = hasattr(dataset, "base") and getattr(dataset.base, "device", None) is not None
    if isinstance(dataset, FewShotWrapper):
        on_gpu = dataset.base.defended.device.type == "cuda"

    kwargs = dict(
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0 if on_gpu else config.num_workers,
        pin_memory=torch.cuda.is_available() and not on_gpu,
    )
    kwargs["collate_fn"] = collate_few_shot
    return DataLoader(dataset, **kwargs)
