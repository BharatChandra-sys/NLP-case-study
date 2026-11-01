import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "01_word2vec": ("word2vec", ["intrinsic_by_fold.csv"]),
    "02_sentiment_classification": ("sentiment", ["scores.csv", "selective.csv", "comparisons.csv"]),
    "03_text_summarization": ("summarization", ["fewsum_scores.csv", "segmentation.csv", "telecom_scores.csv"]),
    "04_factoid_question_answering": ("qa", ["squad_reader_benchmark.csv", "test_ablation.csv", "retrieval.csv"]),
}


def main():
    for name, (topic, files) in FILES.items():
        for item in files:
            new_path = ROOT / "kaggle_outputs" / name / "results" / topic / item
            if not new_path.exists():
                print(name, item, "MISSING")
                continue
            old = pd.read_csv(ROOT / "results" / topic / item).select_dtypes("number")
            new = pd.read_csv(new_path).select_dtypes("number")
            drop = [c for c in old.columns if c in ("seconds",)]
            gap = (old.drop(columns=drop) - new.drop(columns=drop)).abs().max().max()
            print(f"{name} {item} max_abs_diff={gap:.5f}")


if __name__ == "__main__":
    sys.exit(main())
