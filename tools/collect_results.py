import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KERNEL_BASE = "https://www.kaggle.com/code/bharatchandra1234/"
TOPICS = {
    "01_word2vec": ("word2vec", "telelens-01-word2vec", ["intrinsic_summary.csv", "doc_probes.csv", "counterfit_tuning.csv"], ["syn_ant_cosine.png", "word2vec_pca.png"]),
    "02_sentiment_classification": ("sentiment", "telelens-02-sentiment", ["scores.csv", "comparisons.csv", "selective.csv", "robustness.csv"], ["model_f1.png", "risk_coverage.png"]),
    "03_text_summarization": ("summarization", "telelens-03-summarization", ["fewsum_scores.csv", "fewsum_tests.csv", "segmentation.csv", "telecom_scores.csv", "telecom_tests.csv"], ["fewsum_rouge2.png", "telecom_polarity_gap.png"]),
    "04_factoid_question_answering": ("qa", "telelens-04-factoid-qa", ["squad_reader_benchmark.csv", "retrieval.csv", "test_ablation.csv", "test_by_type.csv"], ["qa_ablation.png", "qa_abstention.png"]),
    "05_integrated_system": ("integrated", "telelens-05-integrated", [], []),
}


def table(path):
    frame = pd.read_csv(path).round(4)
    if isinstance(frame.columns[0], str) and frame.columns[0].startswith("Unnamed"):
        frame = frame.rename(columns={frame.columns[0]: ""})
    head = "| " + " | ".join(map(str, frame.columns)) + " |\n|" + "---|" * len(frame.columns) + "\n"
    rows = "".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |\n" for row in frame.values)
    return head + rows


def main():
    for name, (topic, kernel, tables, figures) in TOPICS.items():
        source = ROOT / "kaggle_outputs" / name / "results" / topic
        target = ROOT / "results" / topic
        if not source.exists():
            print("missing", name)
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        text = f"## Recorded Kaggle results\n\nExecuted on a Kaggle T4 GPU as `{kernel}`. Files are in `results/{topic}/`.\n\n"
        for item in tables:
            text += f"**{item}**\n\n{table(target / item)}\n"
        for item in figures:
            text += f"![{item}](../results/{topic}/{item})\n\n"
        if name == "05_integrated_system":
            text += "```json\n" + json.dumps(json.loads((target / "demo.json").read_text(encoding="utf-8")), indent=1)[:5000] + "\n```\n"
            text += "\nUnit tests:\n\n```\n" + (target / "pytest.txt").read_text(encoding="utf-8")[-300:] + "\n```\n"
        path = ROOT / "notebooks" / f"{name}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook["cells"] = [c for c in notebook["cells"] if c.get("id") != "recorded-results"]
        notebook["cells"].append({"cell_type": "markdown", "id": "recorded-results", "metadata": {}, "source": text.splitlines(keepends=True)})
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        print("collected", name)


if __name__ == "__main__":
    main()
