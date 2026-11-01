import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)


def esc(value):
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("$", r"\$")


def num(value, digits=3):
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def write(name, caption, label, spec, header, rows, bold_rows=(), size="small"):
    lines = [r"\begin{table}[H]", r"\centering", rf"\{size}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, row in enumerate(rows):
        cells = [rf"\textbf{{{c}}}" if i in bold_rows else str(c) for c in row]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def word2vec():
    d = pd.read_csv(RES / "word2vec" / "intrinsic_by_fold.csv")
    order = ["SGNS (scratch)", "gensim Word2Vec", "FastText", "SGNS + negation", "PA-W2V"]
    rows = []
    for model in order:
        m = d[d.model == model]
        pm = lambda c: "--" if m[c].isna().all() else f"{m[c].mean():.3f} $\\pm$ {m[c].std():.3f}"
        rows.append([model, f"{m.vocab.mean():.0f}", pm("syn_mean"), pm("ant_mean"), pm("neg_mean"), pm("syn_ant_auc"), pm("neighbour_purity"), pm("simlex_rho")])
    write("w2v_intrinsic", "Word2Vec evaluation across five grouped folds (mean $\\pm$ s.d.). Evaluation word pairs were never used as constraints.", "tab:w2v",
          "lrcccccc", ["Model", "Vocab", "Syn.\\ cos", "Ant.\\ cos", "Neg.\\ cos", "Syn/Ant AUC", "Purity", "SimLex $\\rho$"], rows, bold_rows=[4], size="footnotesize")
    t = pd.read_csv(RES / "word2vec" / "counterfit_tuning.csv")
    write("w2v_tuning", "Counter-fitting grid on the inner validation split of fold 0. The top row became the default.", "tab:w2vtune", "cccccc",
          ["Repel", "Attract", "Anchor", "Val.\\ macro-F1", "Ant.\\ cos", "Syn.\\ cos"],
          [[r.repel, r.attract, r.anchor, num(r.val_macro_f1, 4), num(r.ant_mean), num(r.syn_mean)] for r in t.itertuples()], bold_rows=[0])
    p = pd.read_csv(RES / "word2vec" / "doc_probes.csv")
    write("w2v_probes", "Sentence-level cosine after mean pooling. Averaging dilutes the word-level gain.", "tab:w2vprobe", "p{7.2cm}ccc",
          ["Sentence pair", "SGNS", "FastText", "PA-W2V"],
          [[esc(f"{r.a} / {r.b}"), num(r._3), num(r.FastText), num(r._7)] for r in p.itertuples()], size="footnotesize")
    n = pd.read_csv(RES / "word2vec" / "neighbours.csv")
    n = n[n.word.isin(["reliable", "unreliable", "stable", "helpful", "bill"])]
    write("w2v_neighbours", "Nearest neighbours on the whole-corpus model.", "tab:w2vnn", "llp{9.2cm}", ["Word", "Model", "Top six neighbours"],
          [[esc(r.word), esc(r.model), esc(r.neighbours)] for r in n.itertuples()], size="footnotesize")


def sentiment():
    d = RES / "sentiment"
    s = pd.read_csv(d / "scores.csv")
    write("sent_scores", "Out-of-fold sentiment results on 9,874 reviews (grouped five-fold).", "tab:sent", "lcccccc",
          ["Model", "Accuracy", "95\\% CI", "Macro-F1", "Bad recall", "Bad prec.", "ECE"],
          [[esc(r.model), num(r.accuracy), f"{r.acc_lo:.3f}--{r.acc_hi:.3f}", num(r.macro_f1), num(r.bad_recall), num(r.bad_precision), num(r.ece)] for r in s.itertuples()],
          bold_rows=[0], size="footnotesize")
    c = pd.read_csv(d / "comparisons.csv")
    write("sent_tests", "Paired comparisons: bootstrap macro-F1 difference and exact McNemar test.", "tab:senttest", "p{6.4cm}ccc",
          ["Comparison (A vs B)", "F1 diff", "95\\% CI", "McNemar $p$"],
          [[esc(f"{r.a} vs {r.b}"), f"{r.f1_diff:+.4f}", f"{r.ci_lo:+.4f} to {r.ci_hi:+.4f}", "$<$0.0001" if r.mcnemar_p < 0.0001 else f"{r.mcnemar_p:.4f}"] for r in c.itertuples()], size="footnotesize")
    v = pd.read_csv(d / "selective.csv")
    write("sent_selective", "Selective triage with cross-fitted thresholds: the requested risk is compared with the risk achieved on unseen folds.", "tab:selective", "lccccc",
          ["Model", "Target risk", "Coverage", "Achieved risk", "Deferred", "Bad recall (auto)"],
          [[esc(r.model), num(r.target_risk, 2), num(r.coverage), num(r.risk), r.deferred, num(r.auto_bad_recall)] for r in v.itertuples()], size="footnotesize")
    b = pd.read_csv(d / "robustness.csv")
    write("sent_robust", "Robustness: authored contrast set (30 sentences) and cross-domain test on Comcast complaints.", "tab:robust", "lcccc",
          ["Model (trained on all 10K)", "Contrast acc.", "Comcast macro-F1", "Comcast acc.", "Comcast Bad recall"],
          [[esc(r.model), num(r.contrast_accuracy), num(r.comcast_macro_f1), num(r.comcast_accuracy), num(r.comcast_bad_recall)] for r in b.itertuples()], size="footnotesize")


def summarization():
    d = RES / "summarization"
    f = pd.read_csv(d / "fewsum_scores.csv")
    write("sum_fewsum", "FewSum test (20 products, three human references each): ROUGE F1.", "tab:fewsum", "lccccc",
          ["Method", "Words", "ROUGE-1", "ROUGE-2", "95\\% CI (R-2)", "ROUGE-L"],
          [[esc(r.method), f"{r.words:.1f}", num(r.rouge1), num(r.rouge2), f"{r.rouge2_lo:.3f}--{r.rouge2_hi:.3f}", num(r.rougeL)] for r in f.itertuples()], size="footnotesize")
    s = pd.read_csv(d / "segmentation.csv")
    r2 = json.loads((d / "stripped_rouge2.json").read_text())
    write("sum_seg", f"Sentence-boundary recovery after stripping punctuation from {int(s.reviews.iloc[0])} FewSum reviews (tolerance two words).", "tab:seg", "lcccc",
          ["Segmenter", "Precision", "Recall", "F1", "ROUGE-2 (test)"],
          [[esc(r.method), num(r.precision), num(r.recall), num(r.f1), num(r2[r.method])] for r in s.itertuples()], bold_rows=[2])
    t = pd.read_csv(d / "telecom_scores.csv")
    write("sum_telecom", "Telecom digests over 47 provider and provider-by-aspect clusters (no human references).", "tab:telecom", "lccccc",
          ["Method", "Polarity gap", "Aspect recall", "Redundancy", "Centroid cos", "Words"],
          [[esc(r.method), num(r.polarity_gap), num(r.aspect_recall), num(r.redundancy), num(r.centroid_cosine), f"{r.words:.1f}"] for r in t.itertuples()], bold_rows=[4], size="footnotesize")
    w = pd.read_csv(d / "telecom_tests.csv")
    write("sum_tests", "Paired Wilcoxon tests across the 47 telecom clusters.", "tab:telecomtest", "llccc", ["Metric", "Versus", "PAB-MMR", "Other", "$p$"],
          [[r.metric.replace("_", " ").capitalize(), esc(r.versus), num(r.pab_mean), num(r.other_mean), "$<$0.0001" if r.wilcoxon_p < 0.0001 else f"{r.wilcoxon_p:.4f}"] for r in w.itertuples()])
    e = pd.read_csv(d / "telecom_digests.csv").head(2)
    write("sum_examples", "Generated digests (lowercase source text, segments separated by bars).", "tab:digest", "lp{11.5cm}", ["Cluster", "Summary"],
          [[f"{esc(r.provider)} {name}", esc(getattr(r, attr))] for r in e.itertuples() for name, attr in (("PAB-MMR", "_3"), ("MMR", "MMR"))], size="footnotesize")


def qa():
    d = RES / "qa"
    b = pd.read_csv(d / "squad_reader_benchmark.csv")
    write("qa_squad", "Reader benchmark on a seeded 2,000-question SQuAD v1.1 dev sample with the gold paragraph supplied.", "tab:squad", "lcccc",
          ["Reader", "EM", "95\\% CI", "F1", "Seconds"], [[esc(r.reader), num(r.em), f"{r.em_lo:.3f}--{r.em_hi:.3f}", num(r.f1), f"{r.seconds:.0f}"] for r in b.itertuples()])
    r = pd.read_csv(d / "retrieval.csv")
    r = r[r.split == "test"]
    write("qa_retrieval", "Retrieval of the evidence complaint among 5,629 documents (held-out answerable questions).", "tab:retrieval", "lcccc",
          ["Retriever", "Recall@1", "Recall@5", "Recall@20", "MRR"], [[esc(x.retriever), num(x._3), num(x._4), num(x._5), num(x.mrr)] for x in r.itertuples()], size="footnotesize")
    a = pd.read_csv(d / "test_ablation.csv")
    write("qa_ablation", "Ablation on 68 held-out answerable questions.", "tab:qaablation", "lccc", ["System", "EM", "95\\% CI", "F1"],
          [[esc(x.system), num(x.em), f"{x.em_lo:.3f}--{x.em_hi:.3f}", num(x.f1)] for x in a.itertuples()], bold_rows=[len(a) - 1], size="footnotesize")
    t = pd.read_csv(d / "test_by_type.csv")
    write("qa_types", "Dev-selected system by detected answer type.", "tab:qatype", "lccc", ["Type", "Questions", "EM", "F1"],
          [[x.type, x.questions, num(x.em), num(x.f1)] for x in t.itertuples()])
    s = json.loads((d / "test_selective.json").read_text())
    rows = []
    for name, key in (("Always answer", "no_abstention"), (f"Abstain below $\\tau$ = {s['tau']:.2f}", "with_abstention")):
        v = s[key]
        rows.append([name, num(v["overall_accuracy"]), num(v["answer_coverage"]), num(v["answered_em"]), num(v["answered_f1"]), num(v["unanswerable_rejection"])])
    write("qa_selective", "Abstention on all 88 held-out questions (68 answerable, 20 unanswerable).", "tab:qaabstain", "lccccc",
          ["Policy", "Overall acc.", "Coverage", "Answered EM", "Answered F1", "Rejected"], rows, bold_rows=[1], size="footnotesize")


def integrated():
    demo = json.loads((RES / "integrated" / "demo.json").read_text(encoding="utf-8"))
    write("demo_triage", "Integrated system: live triage output.", "tab:demotriage", "p{8.6cm}ccc", ["Incoming review", "Prediction", "P(Good)", "Decision"],
          [[esc(t["review"]), t["prediction"], num(t["probability_good"]), esc(t["decision"].replace("_", " "))] for t in demo["triage"]], size="footnotesize")
    write("demo_qa", "Integrated system: live question answering with evidence or abstention.", "tab:demoqa", "p{8.0cm}p{2.6cm}cc", ["Question", "Answer", "Score", "Complaint"],
          [[esc(a["question"]), esc(a["answer"] or "(abstained)"), num(a["score"], 2), a.get("doc_id", "--")] for a in demo["answers"]], size="footnotesize")
    (OUT / "demo_digest.tex").write_text(r"\textit{" + esc(" | ".join(demo["digest"]["summary"])) + "}", encoding="utf-8")


if __name__ == "__main__":
    for step in (word2vec, sentiment, summarization, qa, integrated):
        step()
    print("tables written", len(list(OUT.glob("*.tex"))))
