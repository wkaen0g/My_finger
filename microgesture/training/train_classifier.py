"""MLP gesture classifier training pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

from ._hagrid_common import GESTURE_LABELS as _GESTURES

# ── model ─────────────────────────────────────────────────────────────────

class GestureMLP(nn.Module):
    """3-layer MLP for 5-class gesture classification."""

    def __init__(self, input_dim: int = 70, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 5),
        )

    def forward(self, x):
        return self.net(x)


# ── training ──────────────────────────────────────────────────────────────

def load_data(data_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load .npz files from data_dir, return (X, y) tensors."""
    data_dir = Path(data_dir)
    features_list = []
    labels_list = []

    for gesture_idx, name in enumerate(_GESTURES):
        path = data_dir / f"features_{name}.npz"
        if not path.exists():
            logger.warning("Missing %s, skipping", path.name)
            continue
        data = np.load(path)
        feats = data["features"]
        features_list.append(feats)
        labels_list.append(np.full(len(feats), gesture_idx, dtype=np.int64))
        logger.info("Loaded %s: %d samples", name, len(feats))

    if not features_list:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    X = np.concatenate(features_list)
    y = np.concatenate(labels_list)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


def evaluate(model: GestureMLP, test_dir: str | Path,
             device: torch.device | None = None) -> dict:
    """Evaluate a trained model on a separate test set.

    Returns dict with keys: accuracy, per_class (list of (label, acc)),
    confusion (CxC matrix), total_samples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_data(test_dir)
    model.eval()
    model.to(device)
    X, y = X.to(device), y.to(device)

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        correct = (preds == y).float()

    n_total = len(y)
    acc = correct.mean().item()
    n_classes = logits.shape[1]

    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y.cpu(), preds.cpu()):
        confusion[t, p] += 1

    per_class = []
    labels_map = list(_GESTURES)
    for i in range(n_classes):
        mask = (y == i)
        if mask.sum() > 0:
            cls_acc = correct[mask].mean().item()
            per_class.append((labels_map[i] if i < len(labels_map) else f"class_{i}", cls_acc))

    logger.info("=== Test Set Evaluation ===")
    logger.info("Accuracy: %.4f (%.1f%%)", acc, acc * 100)
    for name, cls_acc in per_class:
        logger.info("  %s: %.1f%%", name, cls_acc * 100)

    return {"accuracy": acc, "per_class": per_class, "confusion": confusion,
            "total_samples": n_total}


def train(
    data_dir: str | Path,
    model_dir: str | Path,
    *,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 128,
    dropout: float = 0.3,
    test_dir: str | Path | None = None,
) -> GestureMLP:
    """Train MLP classifier, return the model."""

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    X, y = load_data(data_dir)
    n_total = len(X)
    n_train = int(n_total * 0.8)

    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = GestureMLP(input_dim=70, hidden=hidden, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X).argmax(dim=1)
                correct += (preds == batch_y).sum().item()

        acc = correct / (n_total - n_train)
        avg_loss = total_loss / n_train

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_dir / "best_model.pt")

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info("Epoch %3d | loss=%.4f | val_acc=%.3f (best=%.3f)",
                        epoch + 1, avg_loss, acc, best_acc)

    # Load best
    model.load_state_dict(torch.load(model_dir / "best_model.pt"))
    logger.info("Training complete. Best val_acc=%.3f", best_acc)

    # ── Test set evaluation ────────────────────────────────────────────
    if test_dir and Path(test_dir).exists():
        evaluate(model, test_dir, device)

    return model


def train_with_pretrain(
    pretrain_dir: str | Path | None,
    finetune_dir: str | Path,
    model_dir: str | Path,
    *,
    epochs_pretrain: int = 40,
    epochs_finetune: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    lr_finetune: float = 1e-4,
    hidden: int = 128,
    dropout: float = 0.3,
    test_dir: str | Path | None = None,
) -> GestureMLP:
    """Two-stage training: pretrain on HaGRID, fine-tune on self-collected data."""

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GestureMLP(input_dim=70, hidden=hidden, dropout=dropout).to(device)

    # Stage 1: pretrain
    if pretrain_dir and Path(pretrain_dir).exists():
        logger.info("=== Stage 1: Pretrain on %s ===", pretrain_dir)
        X, y = load_data(pretrain_dir)
        n_train = int(len(X) * 0.8)
        train_ds = TensorDataset(X[:n_train], y[:n_train])
        val_ds = TensorDataset(X[n_train:], y[n_train:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        opt = optim.Adam(model.parameters(), lr=lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_pretrain)
        best_acc = _run_epochs(model, train_loader, val_loader, opt, sched,
                               epochs_pretrain, device, model_dir, "pretrain_best.pt")
        logger.info("Pretrain complete. Best val_acc=%.3f", best_acc)
        model.load_state_dict(torch.load(model_dir / "pretrain_best.pt"))

    # Stage 2: fine-tune on self-collected data
    logger.info("=== Stage 2: Fine-tune on %s ===", finetune_dir)
    X, y = load_data(finetune_dir)
    n_train = int(len(X) * 0.8)
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    opt = optim.Adam(model.parameters(), lr=lr_finetune)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_finetune)
    best_acc = _run_epochs(model, train_loader, val_loader, opt, sched,
                           epochs_finetune, device, model_dir, "best_model.pt")
    logger.info("Fine-tune complete. Best val_acc=%.3f", best_acc)
    model.load_state_dict(torch.load(model_dir / "best_model.pt"))

    # ── Test set evaluation ────────────────────────────────────────────
    if test_dir and Path(test_dir).exists():
        evaluate(model, test_dir, device)

    return model


def _run_epochs(model, train_loader, val_loader, optimizer, scheduler,
                epochs: int, device, model_dir: Path, save_name: str) -> float:
    """Shared training loop. Returns best val_acc."""
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    n_train = sum(len(batch[0]) for batch in train_loader)
    n_val = sum(len(batch[0]) for batch in val_loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                correct += (model(batch_X).argmax(dim=1) == batch_y).sum().item()

        acc = correct / n_val
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_dir / save_name)

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info("Epoch %3d | loss=%.4f | val_acc=%.3f (best=%.3f)",
                        epoch + 1, total_loss / n_train, acc, best_acc)

    return best_acc
