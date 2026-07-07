"""Model test harness — evaluate ONNX / PyTorch model on any dataset.

Usage:
  python -m microgesture.training.model_test [--model onnx|torch] [--data-dir ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def test_model(
    model_path: str | Path,
    data_dir: str | Path,
    *,
    model_type: str = "onnx",
) -> dict:
    """Evaluate a model on a labeled dataset. Returns metrics dict.

    Args:
        model_path: Path to ONNX file or PyTorch .pt checkpoint.
        data_dir: Directory containing features_*.npz files.
        model_type: "onnx" or "torch".

    Returns:
        dict with keys: accuracy, per_class, confusion, sklearn_report,
                        total, correct, wrong_indices.
    """
    from .train_classifier import load_data, GestureMLP, NUM_CLASSES
    from ._hagrid_common import GESTURE_LABELS

    labels = list(GESTURE_LABELS)

    # ── Load model ─────────────────────────────────────────────────────
    X, y = load_data(data_dir)
    n_total = len(y)

    if model_type == "onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        def predict(batch_x):
            out = session.run([output_name], {input_name: batch_x.numpy().astype(np.float32)})[0]
            return torch.from_numpy(out)

    elif model_type == "torch":
        model = GestureMLP(input_dim=70)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        def predict(batch_x):
            with torch.no_grad():
                return model(batch_x)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ── Inference ──────────────────────────────────────────────────────
    all_preds = []
    all_confs = []
    batch_size = 256

    for i in range(0, n_total, batch_size):
        batch_x = X[i:i + batch_size]
        logits = predict(batch_x)
        exps = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
        probs = exps / exps.sum(dim=1, keepdim=True)
        preds = probs.argmax(dim=1)
        confs = probs[range(len(preds)), preds]
        all_preds.append(preds)
        all_confs.append(confs)

    preds = torch.cat(all_preds)
    confs = torch.cat(all_confs)
    correct = (preds == y).float()

    # ── Metrics ────────────────────────────────────────────────────────
    acc = correct.mean().item()

    # Confusion matrix
    n_classes = len(labels)
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y, preds):
        confusion[t, p] += 1

    # Per-class precision/recall/f1
    per_class = []
    for i in range(n_classes):
        tp = confusion[i, i].item()
        fn = confusion[i].sum().item() - tp
        fp = confusion[:, i].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = confusion[i].sum().item()
        per_class.append({
            "label": labels[i] if i < len(labels) else f"class_{i}",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "accuracy": tp / support if support > 0 else 0.0,
        })

    wrong_indices = (correct == 0).nonzero(as_tuple=True)[0].tolist()

    return {
        "accuracy": acc,
        "total": n_total,
        "correct": int(correct.sum().item()),
        "wrong": n_total - int(correct.sum().item()),
        "confusion": confusion,
        "per_class": per_class,
        "wrong_indices": wrong_indices,
    }


def print_report(metrics: dict) -> None:
    """Pretty-print evaluation results."""
    print()
    print("=" * 52)
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Correct:  {metrics['correct']} / {metrics['total']}")
    print(f"  Wrong:    {metrics['wrong']}")
    print("=" * 52)

    # Per-class table
    print(f"\n{'Class':<16} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Acc':>6} {'N':>6}")
    print("-" * 52)
    for c in metrics["per_class"]:
        print(f"{c['label']:<16} {c['precision']:6.1%} {c['recall']:6.1%} "
              f"{c['f1']:6.1%} {c['accuracy']:6.1%} {c['support']:>6}")

    # Confusion matrix
    cm = metrics["confusion"]
    n = cm.shape[0]
    labels = [c["label"][:4] for c in metrics["per_class"]]

    print(f"\nConfusion Matrix:")
    header = "     " + " ".join(f"{l:>5}" for l in labels)
    print(header)
    for i in range(n):
        row = " ".join(f"{v:5}" for v in cm[i].tolist())
        print(f"  {labels[i]:>3} {row}")

    # Worst errors
    wrong = metrics["wrong_indices"]
    if wrong and len(wrong) <= 20:
        print(f"\nMisclassified samples (n={len(wrong)}):")
        for idx in wrong[:10]:
            print(f"  sample {idx}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test gesture classifier model")
    parser.add_argument("--model", default="onnx", choices=["onnx", "torch"],
                        help="Model type (default: onnx)")
    parser.add_argument("--model-path", default=None,
                        help="Model file path (default: training/models/best_model.pt "
                             "or models/classifier.onnx)")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with features_*.npz files")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Auto-detect model path
    if args.model_path is None:
        if args.model == "onnx":
            args.model_path = "microgesture/models/classifier.onnx"
        else:
            args.model_path = "microgesture/training/models/best_model.pt"

    if not Path(args.model_path).exists():
        print(f"Model not found: {args.model_path}", file=sys.stderr)
        return 1

    metrics = test_model(args.model_path, args.data_dir, model_type=args.model)
    print_report(metrics)
    return 0 if metrics["wrong"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
