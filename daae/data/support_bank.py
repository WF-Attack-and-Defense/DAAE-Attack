"""Per-class support memory for few-shot inference."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from daae.config import DAAEConfig
from daae.data.cache import StackedPairedDataset, preload_pairs_parallel
from daae.data.few_shot import build_label_index
from daae.recover_utils import ClassPrototypeCache, build_prototype_cache


def find_train_cache(
    cache_dir: Union[str, Path],
    preferred: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Locate a ``train_*.npz`` cache under ``cache_dir`` (or an explicit file)."""
    if preferred is not None:
        p = Path(preferred)
        if p.is_file():
            return p
        raise FileNotFoundError(f"--support-cache not found: {p}")

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return None
    candidates = sorted(
        cache_dir.glob("train_*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


class ClassSupportBank:
    """
    Stores K defended traces + dummy masks per class from few-shot training data.

    Used at inference to remove dummy packets from defended traces without
    requiring the original trace at test time.
    """

    def __init__(
        self,
        defended: torch.Tensor,
        dummy_mask: torch.Tensor,
        labels: torch.Tensor,
        support_shots: int,
        label_index: Dict[int, List[int]],
        defended_len: Optional[torch.Tensor] = None,
    ):
        self.defended = defended
        self.dummy_mask = dummy_mask
        self.labels = labels
        self.support_shots = support_shots
        self.label_index = label_index
        self.defended_len = defended_len
        self._device: Optional[torch.device] = None
        self._class_support: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._prototype_cache: Dict[int, ClassPrototypeCache] = {}

    @classmethod
    def from_stacked(
        cls,
        dataset: StackedPairedDataset,
        support_shots: int,
    ) -> "ClassSupportBank":
        labels = dataset.labels.cpu().numpy()
        bank = cls(
            dataset.defended,
            dataset.dummy_mask,
            dataset.labels,
            support_shots,
            build_label_index(labels),
            defended_len=getattr(dataset, "defended_len", None),
        )
        bank.build_caches()
        return bank

    @classmethod
    def from_pairs(
        cls,
        pairs: List,
        config: DAAEConfig,
        cache_dir,
        device: Optional[torch.device] = None,
    ) -> "ClassSupportBank":
        arrays = preload_pairs_parallel(
            pairs,
            config.max_len,
            config.trace_format,
            config.preload_workers,
            split_name="support_bank",
            cache_dir=cache_dir,
            use_cache=config.use_cache,
            alignment_method=config.alignment_method,
        )
        ds = StackedPairedDataset(arrays, device=device)
        return cls.from_stacked(ds, config.support_shots)

    @classmethod
    def from_train_cache(
        cls,
        cache_path: Union[str, Path],
        support_shots: int,
        device: Optional[torch.device] = None,
    ) -> "ClassSupportBank":
        """
        Build a support bank from a training ``.npz`` cache.

        Keep masks were labeled during training (with originals) and stored in
        the cache, so recovery does not need originals at inference time.
        """
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Train cache not found: {cache_path}")
        print(f"Loading support bank from train cache: {cache_path}", flush=True)
        arrays = dict(np.load(cache_path))
        required = ("defended", "dummy_mask", "labels", "defended_len")
        missing = [k for k in required if k not in arrays]
        if missing:
            raise KeyError(f"Train cache missing keys {missing}: {cache_path}")
        ds = StackedPairedDataset(arrays, device=device)
        return cls.from_stacked(ds, support_shots)

    def build_caches(self) -> None:
        """Precompute per-class GPU support tensors and CPU decode prototypes."""
        self._class_support.clear()
        self._prototype_cache.clear()
        for cls_id, pool in self.label_index.items():
            k = min(self.support_shots, len(pool))
            ids = pool[:k]
            s_def = self.defended[ids]
            s_mask = self.dummy_mask[ids]
            self._class_support[int(cls_id)] = (s_def, s_mask)

            dirs = s_def[:, 0].cpu().numpy()
            masks = s_mask.cpu().numpy()
            if self.defended_len is not None:
                lens = self.defended_len[ids].cpu().numpy()
            else:
                lens = np.full(k, s_mask.shape[-1], dtype=np.int32)
            self._prototype_cache[int(cls_id)] = build_prototype_cache(dirs, masks, lens)

    def ensure_on_device(self, device: torch.device) -> None:
        if self._device == device:
            return
        self.defended = self.defended.to(device, non_blocking=True)
        self.dummy_mask = self.dummy_mask.to(device, non_blocking=True)
        if self.defended_len is not None:
            self.defended_len = self.defended_len.to(device, non_blocking=True)
        self._device = device
        self._class_support.clear()
        for cls_id, pool in self.label_index.items():
            k = min(self.support_shots, len(pool))
            ids = pool[:k]
            self._class_support[int(cls_id)] = (
                self.defended[ids],
                self.dummy_mask[ids],
            )

    def get(self, label: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (1, K, C, L) defended and (1, K, L) masks for a class."""
        self.ensure_on_device(device)
        pool = self.label_index.get(int(label), [])
        if not pool:
            raise KeyError(f"No support traces for class {label}")
        s_def, s_mask = self._class_support[int(label)]
        return s_def.unsqueeze(0), s_mask.unsqueeze(0)

    def get_prototype_cache(self, label: int) -> ClassPrototypeCache:
        return self._prototype_cache[int(label)]

    def attach_to_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Add per-sample support tensors for a batch."""
        labels = batch["label"].tolist()
        s_defs, s_masks = [], []
        for label in labels:
            sd, sm = self.get(label, device)
            s_defs.append(sd.squeeze(0))
            s_masks.append(sm.squeeze(0))
        batch = dict(batch)
        batch["support_defended"] = torch.stack(s_defs, dim=0)
        batch["support_mask"] = torch.stack(s_masks, dim=0)
        return batch

    def expand_support(
        self, label: int, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (B, K, C, L) and (B, K, L) by tiling class support."""
        s_def, s_mask = self.get(label, device)
        return (
            s_def.repeat(batch_size, 1, 1, 1),
            s_mask.repeat(batch_size, 1, 1),
        )
