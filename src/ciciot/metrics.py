from __future__ import annotations

import math

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[float, float, float]:
    per_class = per_class_precision_recall_f1(y_true, y_pred, num_classes)
    precisions = [values["precision"] for values in per_class.values()]
    recalls = [values["recall"] for values in per_class.values()]
    f1s = [values["f1"] for values in per_class.values()]
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def per_class_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, dict[str, float]]:
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    results: dict[str, dict[str, float]] = {}

    for idx in range(num_classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        results[str(idx)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[idx, :].sum()),
        }

    return results


def multiclass_macro_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    aucs = []
    for class_idx in range(num_classes):
        binary_true = (y_true == class_idx).astype(np.int32)
        scores = y_prob[:, class_idx]
        positives = binary_true.sum()
        negatives = len(binary_true) - positives
        if positives == 0 or negatives == 0:
            continue
        aucs.append(binary_auc(binary_true, scores))

    if not aucs:
        return math.nan
    return float(np.mean(aucs))


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    positive_mask = y_true == 1
    n_pos = positive_mask.sum()
    n_neg = len(y_true) - n_pos
    rank_sum = ranks[positive_mask].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_classification(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = y_prob.shape[1]
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, num_classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "auc_macro_ovr": multiclass_macro_auc(y_true, y_prob, num_classes),
    }


def evaluate_classification_detailed(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, object]:
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = y_prob.shape[1]
    summary = evaluate_classification(y_true, y_prob)
    summary["confusion_matrix"] = confusion_matrix(y_true, y_pred, num_classes).tolist()
    summary["per_class"] = per_class_precision_recall_f1(y_true, y_pred, num_classes)
    return summary
