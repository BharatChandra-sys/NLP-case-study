from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "sentiment"


def main():
    d = pd.read_csv(RES / "oof_probabilities.csv")
    y = d["label"].to_numpy()
    rows, confusion, queue = [], [], []
    for model in [c for c in d.columns if c not in ("row", "label", "fold")]:
        p = d[model].to_numpy()
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        folds = [f1_score(y[d.fold == k], pred[d.fold == k], average="macro") for k in sorted(d.fold.unique())]
        rows.append({"model": model, "roc_auc": roc_auc_score(y, p), "pr_auc_bad": average_precision_score(1 - y, 1 - p), "mcc": matthews_corrcoef(y, pred),
                     "kappa": cohen_kappa_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred), "recall_good": tp / (tp + fn),
                     "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "fold_f1_mean": np.mean(folds), "fold_f1_std": np.std(folds)})
        confusion.append({"model": model, "bad_as_bad": tn, "bad_as_good": fp, "good_as_bad": fn, "good_as_good": tp})
        order = np.argsort(p)
        row = {"model": model}
        for k in (100, 500, 1000, 2000):
            row[f"bad_hit@{k}"] = float((y[order[:k]] == 0).mean())
        row["bad_recall@2000"] = float((y[order[:2000]] == 0).sum() / (y == 0).sum())
        row["first_good_rank"] = int(np.argmax(y[order] == 1)) + 1
        queue.append(row)
    frame = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    frame.to_csv(RES / "extra_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(RES / "confusion_all.csv", index=False)
    pd.DataFrame(queue).to_csv(RES / "triage_queue_hits.csv", index=False)
    print(frame.round(4).to_string())
    print(pd.DataFrame(confusion).to_string())
    print(pd.DataFrame(queue).round(4).to_string())


if __name__ == "__main__":
    main()
