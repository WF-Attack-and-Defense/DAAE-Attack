"""Evaluate DAAE recovery quality and website fingerprinting attack accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from daae.cli import add_dataset_scope_arguments, add_path_arguments, add_alignment_arguments, resolve_paths
from daae.config import DAAEConfig
from daae.data.alignment import align_traces
from daae.data.cell_io import discover_trace_pairs, parse_cell_file
from daae.data.dataset import split_pairs_by_class
from daae.data.features import extract_directions, extract_trace_features
from daae.models.classifier import WFClassifier, classifier_loss
from daae.models.factory import load_model_from_checkpoint as load_model


class TraceLabelDataset(Dataset):
    """Single-channel traces with class labels for WF classification."""

    def __init__(
        self,
        pairs: List[Tuple[Path, Path, int, int, int]],
        max_len: int,
        source: str = "defended",
        recovered_dir: Path | None = None,
        trace_format: str = "direction_timing",
    ):
        self.pairs = pairs
        self.max_len = max_len
        self.source = source
        self.recovered_dir = recovered_dir
        self.trace_format = trace_format

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        orig_path, def_path, label_idx, inst_id, orig_cls_id = self.pairs[idx]
        if self.source == "original":
            trace = parse_cell_file(orig_path)
        elif self.source == "recovered":
            assert self.recovered_dir is not None
            rec_path = self.recovered_dir / f"{orig_cls_id}-{inst_id}.cell"
            trace = parse_cell_file(rec_path if rec_path.exists() else def_path)
        else:
            trace = parse_cell_file(def_path)

        features, valid_len = extract_trace_features(trace, self.max_len, self.trace_format)
        # WF classifier uses direction channel (Tor WF standard)
        feat = features[0]
        padded = np.zeros(self.max_len, dtype=np.float32)
        padded[:valid_len] = feat[:valid_len]
        return {
            "trace": torch.from_numpy(padded),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> Tuple[WFClassifier, Dict[str, float]]:
    model = WFClassifier(num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    best_acc = 0.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["trace"])
            loss, _ = classifier_loss(logits, batch["label"])
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["trace"])
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["label"]).sum().item()
                total += batch["label"].numel()
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_val_acc": best_acc}


@torch.no_grad()
def test_classifier(model: WFClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["trace"])
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].numel()
    return correct / max(total, 1)


def compute_recovery_metrics(
    pairs: List[Tuple[Path, Path, int, int]],
    recovered_dir: Path,
    trace_format: str,
    alignment_method: str = "hamming",
) -> Dict[str, float]:
    dir_matches, length_ratios = [], []
    for orig_path, _, _, inst_id in pairs:
        cls_id = int(orig_path.stem.split("-")[0])
        rec_path = recovered_dir / f"{cls_id}-{inst_id}.cell"
        if not rec_path.exists():
            continue
        orig = parse_cell_file(orig_path)
        rec = parse_cell_file(rec_path)
        orig_dirs = extract_directions(orig)
        rec_dirs = extract_directions(rec)
        align = align_traces(orig_dirs, rec_dirs, method=alignment_method)
        dir_matches.append(align.match_ratio)
        length_ratios.append(len(rec_dirs) / max(len(orig_dirs), 1))

    if not dir_matches:
        return {"direction_match": 0.0, "length_ratio": 0.0, "alignment_score": 0.0}

    return {
        "direction_match": float(np.mean(dir_matches)),
        "length_ratio": float(np.mean(length_ratios)),
        "alignment_score": float(np.mean(dir_matches)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DAAE and WF attack")
    parser.add_argument("--checkpoint", required=True, type=Path)
    add_path_arguments(parser)
    add_dataset_scope_arguments(parser)
    add_alignment_arguments(parser, default=None)
    parser.add_argument("--clf-epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    original_root, defended_root, output_dir = resolve_paths(args)
    recovered_dir = output_dir / "recovered"

    device = torch.device(args.device)
    _, config = load_model(str(args.checkpoint.resolve()), device)
    config.num_classes = args.num_classes
    config.num_instances = args.num_instances
    if args.alignment_method is not None:
        config.alignment_method = args.alignment_method

    pairs = discover_trace_pairs(
        original_root, defended_root, config.num_classes, config.num_instances
    )
    split = split_pairs_by_class(pairs, config.train_ratio, config.val_ratio, config.seed)
    classes = sorted({p[2] for p in pairs})
    num_classes = len(classes)
    label_map = {c: i for i, c in enumerate(classes)}

    def remap_pairs(items):
        return [(a, b, label_map[c], d, c) for a, b, c, d in items]

    train_pairs = remap_pairs(split.train)
    val_pairs = remap_pairs(split.val)
    test_pairs = remap_pairs(split.test)

    recovery_metrics = compute_recovery_metrics(
        split.test, recovered_dir, config.trace_format, config.alignment_method
    )
    print(f"Recovery metrics: {recovery_metrics}")

    results = {"recovery": recovery_metrics, "attack_accuracy": {}}

    for source in ["defended", "recovered", "original"]:
        train_ds = TraceLabelDataset(
            train_pairs, config.max_len, source=source,
            recovered_dir=recovered_dir, trace_format=config.trace_format,
        )
        val_ds = TraceLabelDataset(
            val_pairs, config.max_len, source=source,
            recovered_dir=recovered_dir, trace_format=config.trace_format,
        )
        test_ds = TraceLabelDataset(
            test_pairs, config.max_len, source=source,
            recovered_dir=recovered_dir, trace_format=config.trace_format,
        )
        train_loader = DataLoader(train_ds, batch_size=config.clf_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.clf_batch_size)
        test_loader = DataLoader(test_ds, batch_size=config.clf_batch_size)

        clf, _ = train_classifier(
            train_loader, val_loader, num_classes, device, epochs=args.clf_epochs
        )
        acc = test_classifier(clf, test_loader, device)
        results["attack_accuracy"][source] = acc
        print(f"WF attack accuracy ({source}): {acc:.4f}")

    out_path = output_dir / "evaluation.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation to {out_path}")


if __name__ == "__main__":
    main()
