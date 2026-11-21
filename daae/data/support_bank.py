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


def _slot_signature(dirs: np.ndarray, n_slots: int = 32) -> np.ndarray:
    """Mean direction in equal-length windows (defended-domain class signature)."""
    dirs = np.asarray(dirs, dtype=np.float32).reshape(-1)
    sig = np.zeros(n_slots, dtype=np.float32)
    n = len(dirs)
    if n == 0:
        return sig
    idx = np.clip((np.arange(n) * n_slots / max(n, 1)).astype(np.int64), 0, n_slots - 1)
    counts = np.bincount(idx, minlength=n_slots).astype(np.float32)
    sums = np.bincount(idx, weights=dirs, minlength=n_slots).astype(np.float32)
    nz = counts > 0
    sig[nz] = sums[nz] / counts[nz]
    return sig


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
        original_len: Optional[torch.Tensor] = None,
    ):
        self.defended = defended
        self.dummy_mask = dummy_mask
        self.labels = labels
        self.support_shots = support_shots
        self.label_index = label_index
        self.defended_len = defended_len
        self.original_len = original_len
        self._device: Optional[torch.device] = None
        self._class_support: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._prototype_cache: Dict[int, ClassPrototypeCache] = {}
        self._global_support: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._mixed_packs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._def_slot: Dict[int, np.ndarray] = {}
        self._median_def_len: Dict[int, int] = {}
        self._median_orig_len: Dict[int, int] = {}
        self._keep_ratio: Dict[int, float] = {}
        self._global_median_orig: int = 1000
        self._global_keep_ratio: float = 0.35

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
            original_len=getattr(dataset, "original_len", None),
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
        self._global_support = None
        self._mixed_packs = None
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
        self._build_infer_index()

    def _build_infer_index(self) -> None:
        """Train-only defended signatures and length priors (no query labels)."""
        self._def_slot.clear()
        self._median_def_len.clear()
        self._median_orig_len.clear()
        self._keep_ratio.clear()
        orig_all: List[int] = []
        for cls_id, pool in self.label_index.items():
            sigs: List[np.ndarray] = []
            def_lens: List[int] = []
            orig_lens: List[int] = []
            for idx in pool:
                if self.defended_len is not None:
                    sl = max(int(self.defended_len[idx].item()), 1)
                else:
                    sl = int(self.defended.shape[-1])
                dirs = self.defended[idx, 0, :sl].detach().cpu().numpy()
                sigs.append(_slot_signature(dirs))
                def_lens.append(sl)
                if self.original_len is not None:
                    orig_lens.append(max(int(self.original_len[idx].item()), 1))
                else:
                    orig_lens.append(
                        max(int(self.dummy_mask[idx, :sl].sum().item()), 1)
                    )
            self._def_slot[int(cls_id)] = np.mean(np.stack(sigs, axis=0), axis=0)
            self._median_def_len[int(cls_id)] = int(np.median(def_lens))
            self._median_orig_len[int(cls_id)] = int(np.median(orig_lens))
            self._keep_ratio[int(cls_id)] = float(np.median(orig_lens)) / float(
                max(np.median(def_lens), 1.0)
            )
            orig_all.extend(orig_lens)
        self._global_median_orig = int(np.median(orig_all)) if orig_all else 1000
        ratios = list(self._keep_ratio.values())
        self._global_keep_ratio = float(np.mean(ratios)) if ratios else 0.35

    def infer_support_class(self, dirs: np.ndarray, def_len: int) -> int:
        """Guess a train class from the defended trace only (filename unused)."""
        if not self._def_slot:
            self._build_infer_index()
        q = _slot_signature(np.asarray(dirs[: max(int(def_len), 0)], dtype=np.float32))
        qn = float(np.linalg.norm(q))
        best_cls, best_score = 0, -1e9
        for cls_id, ref in self._def_slot.items():
            denom = float(np.linalg.norm(ref) * qn)
            cos = float(np.dot(ref, q) / denom) if denom > 0 else 0.0
            med = max(self._median_def_len.get(cls_id, 1), 1)
            rel = abs(int(def_len) - med) / float(med)
            score = cos - 0.35 * rel
            if score > best_score:
                best_score = score
                best_cls = int(cls_id)
        return int(best_cls)

    def target_keep_count(self, cls_id: int, def_len: int) -> int:
        """Length budget from the inferred class prior, clamped to the query."""
        def_len = max(int(def_len), 1)
        orig = int(self._median_orig_len.get(int(cls_id), self._global_median_orig))
        ratio = float(self._keep_ratio.get(int(cls_id), 0.25))
        from_ratio = int(round(ratio * def_len))
        k = int(round(0.7 * orig + 0.3 * from_ratio))
        return max(1, min(def_len, k))

    def ensure_on_device(self, device: torch.device) -> None:
        if self._device == device:
            return
        self.defended = self.defended.to(device, non_blocking=True)
        self.dummy_mask = self.dummy_mask.to(device, non_blocking=True)
        if self.defended_len is not None:
            self.defended_len = self.defended_len.to(device, non_blocking=True)
        self._device = device
        self._class_support.clear()
        self._global_support = None
        self._mixed_packs = None
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

    def attach_global_to_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Attach the same mixed-class dummy prior to every query (no labels)."""
        b = int(batch["defended"].shape[0])
        sd, sm = self.expand_global_support(b, device)
        batch = dict(batch)
        batch["support_defended"] = sd
        batch["support_mask"] = sm
        return batch

    def mixed_support_packs(
        self,
        n_packs: int,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Disjoint mixed-class support packs for a class-agnostic logit ensemble."""
        self.ensure_on_device(device)
        n_packs = max(1, int(n_packs))
        if self._mixed_packs is None or len(self._mixed_packs) != n_packs:
            classes = sorted(self.label_index)
            packs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for p in range(n_packs):
                ids: List[int] = []
                offset = 0
                while len(ids) < self.support_shots and classes:
                    progressed = False
                    for j, cls in enumerate(classes):
                        if j % n_packs != p:
                            continue
                        pool = self.label_index[cls]
                        if offset < len(pool):
                            ids.append(int(pool[offset]))
                            progressed = True
                            if len(ids) >= self.support_shots:
                                break
                    if not progressed:
                        offset += 1
                        if offset > 8:
                            break
                    else:
                        offset += 1
                if not ids:
                    ids = self._global_support_ids()
                idx = torch.tensor(ids, dtype=torch.long, device=self.defended.device)
                packs.append((self.defended[idx], self.dummy_mask[idx]))
            self._mixed_packs = packs
        return [(d.unsqueeze(0), m.unsqueeze(0)) for d, m in self._mixed_packs]

    def expand_support(
        self, label: int, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (B, K, C, L) and (B, K, L) by tiling class support."""
        s_def, s_mask = self.get(label, device)
        return (
            s_def.repeat(batch_size, 1, 1, 1),
            s_mask.repeat(batch_size, 1, 1),
        )

    def _global_support_ids(self) -> List[int]:
        """Round-robin train traces across classes so no query class is favored."""
        ids: List[int] = []
        classes = sorted(self.label_index)
        if not classes:
            return ids
        offset = 0
        while len(ids) < self.support_shots:
            progressed = False
            for cls in classes:
                pool = self.label_index[cls]
                if offset < len(pool):
                    ids.append(int(pool[offset]))
                    progressed = True
                    if len(ids) >= self.support_shots:
                        return ids
            if not progressed:
                break
            offset += 1
        return ids

    def global_keep_ratio(self) -> float:
        """Train-only fraction of original packets (defense dummy rate, no query labels)."""
        if self._keep_ratio:
            return float(self._global_keep_ratio)
        return 0.35

    def get_global(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Class-agnostic dummy-pattern supports, identical for every query."""
        self.ensure_on_device(device)
        if self._global_support is None:
            ids = self._global_support_ids()
            if not ids:
                raise KeyError("No train traces for global supports")
            idx = torch.tensor(ids, dtype=torch.long, device=self.defended.device)
            self._global_support = (self.defended[idx], self.dummy_mask[idx])
        s_def, s_mask = self._global_support
        return s_def.unsqueeze(0), s_mask.unsqueeze(0)

    def expand_global_support(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tile the same mixed-class dummy prior across a batch."""
        s_def, s_mask = self.get_global(device)
        return (
            s_def.repeat(batch_size, 1, 1, 1),
            s_mask.repeat(batch_size, 1, 1),
        )
