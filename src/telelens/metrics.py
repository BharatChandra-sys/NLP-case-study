import re
import string
from collections import Counter

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_report(y_true, y_pred):
    bad = precision_recall_fscore_support(y_true, y_pred, labels=[0], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "bad_precision": float(bad[0][0]),
        "bad_recall": float(bad[1][0]),
        "bad_f1": float(bad[2][0]),
    }


def bootstrap_ci(values, n=2000, seed=42, stat=np.mean):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_macro_f1_diff(y_true, pred_a, pred_b, n=1000, seed=42):
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs.append(f1_score(y_true[idx], pred_a[idx], average="macro") - f1_score(y_true[idx], pred_b[idx], average="macro"))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar_exact(y_true, pred_a, pred_b):
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_only = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    b_only = int(((pred_a != y_true) & (pred_b == y_true)).sum())
    if a_only + b_only == 0:
        return a_only, b_only, 1.0
    return a_only, b_only, float(binomtest(a_only, a_only + b_only, 0.5).pvalue)


def expected_calibration_error(y_true, prob, bins=10):
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    confidence = np.maximum(prob, 1 - prob)
    correct = ((prob >= 0.5).astype(int) == y_true).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence <= hi if hi == 1.0 else confidence < hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def risk_coverage(y_true, prob, grid=None):
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    confidence = np.maximum(prob, 1 - prob)
    correct = (prob >= 0.5).astype(int) == y_true
    grid = np.linspace(0.5, 0.995, 100) if grid is None else grid
    rows = []
    for tau in grid:
        kept = confidence >= tau
        if kept.sum():
            rows.append({"tau": float(tau), "coverage": float(kept.mean()), "risk": float(1 - correct[kept].mean())})
    return rows


def threshold_for_risk(y_true, prob, target_risk):
    rows = risk_coverage(y_true, prob)
    valid = [r for r in rows if r["risk"] <= target_risk]
    return max(valid, key=lambda r: r["coverage"])["tau"] if valid else 1.01


def js_binary(p, q):
    return float(jensenshannon([p, 1 - p], [q, 1 - q], base=2) ** 2)


def normalize_answer(text):
    text = "".join(ch for ch in str(text).lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, answers):
    return float(any(normalize_answer(prediction) == normalize_answer(a) for a in answers))


def token_f1(prediction, answers):
    best = 0.0
    pred = normalize_answer(prediction).split()
    for answer in answers:
        gold = normalize_answer(answer).split()
        common = sum((Counter(pred) & Counter(gold)).values())
        if common:
            p, r = common / len(pred), common / len(gold)
            best = max(best, 2 * p * r / (p + r))
    return best


def retrieval_scores(ranks, ks=(1, 3, 5, 10, 20)):
    ranks = [r if r is not None else np.inf for r in ranks]
    scores = {f"hit@{k}": float(np.mean([r <= k for r in ranks])) for k in ks}
    scores["mrr"] = float(np.mean([0.0 if np.isinf(r) else 1.0 / r for r in ranks]))
    scores["ndcg@10"] = float(np.mean([1.0 / np.log2(r + 1) if r <= 10 else 0.0 for r in ranks]))
    return scores


def rejection_scores(items, tau):
    positives = [not i["answerable"] for i in items]
    rejected = [i["score"] < tau for i in items]
    tp = sum(p and r for p, r in zip(positives, rejected))
    precision = tp / max(sum(rejected), 1)
    recall = tp / max(sum(positives), 1)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-9)}


def selective_qa(items, tau):
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]
    answered = [i for i in answerable if i["score"] >= tau]
    correct = sum(i["em"] for i in answered)
    false_answers = sum(i["score"] >= tau for i in unanswerable)
    return {
        "tau": float(tau),
        "overall_accuracy": (correct + len(unanswerable) - false_answers) / max(len(items), 1),
        "answer_coverage": len(answered) / max(len(answerable), 1),
        "answered_em": correct / max(len(answered), 1),
        "answered_f1": float(np.mean([i["f1"] for i in answered])) if answered else 0.0,
        "unanswerable_rejection": 1 - false_answers / max(len(unanswerable), 1),
    }
