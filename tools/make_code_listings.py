import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report" / "generated" / "code"
OUT.mkdir(parents=True, exist_ok=True)
LISTINGS = {
    "bind_negation": ("text.py", "bind_negation"),
    "skipgram_pairs": ("embeddings.py", "_skipgram_pairs"),
    "train_sgns": ("embeddings.py", "train_sgns"),
    "log_odds": ("embeddings.py", "log_odds_polarity"),
    "polar_constraints": ("embeddings.py", "polar_neighbour_constraints"),
    "counterfit": ("embeddings.py", "polarity_counterfit"),
    "fit_temperature": ("sentiment.py", "fit_temperature"),
    "finetune": ("sentiment.py", "distilbert_finetune"),
    "risk_coverage": ("metrics.py", "risk_coverage"),
    "threshold": ("metrics.py", "threshold_for_risk"),
    "pab_mmr": ("summarize.py", "pab_mmr"),
    "segment_many": ("punctuate.py", "PunctuationRestorer.segment_many"),
    "expand_aspects": ("summarize.py", "expand_aspects"),
    "answer_type": ("qa.py", "answer_type"),
    "rank": ("qa.py", "HybridRetriever.rank"),
    "answer": ("qa.py", "QAPipeline.answer"),
    "selective_qa": ("metrics.py", "selective_qa"),
    "retrieval_scores": ("metrics.py", "retrieval_scores"),
    "triage": ("pipeline.py", "TeleLens.triage"),
}


def find(tree, name):
    node = tree
    for part in name.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    return node


def main():
    for out, (filename, name) in LISTINGS.items():
        source = (ROOT / "src" / "telelens" / filename).read_text(encoding="utf-8")
        node = find(ast.parse(source), name)
        lines = source.splitlines()[node.lineno - 1 : node.end_lineno]
        indent = len(lines[0]) - len(lines[0].lstrip())
        (OUT / f"{out}.py").write_text("\n".join(line[indent:] for line in lines) + "\n", encoding="utf-8")
    print("listings", len(LISTINGS))


if __name__ == "__main__":
    main()
