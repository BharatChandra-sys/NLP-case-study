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
    write("qa_retrieval", "Retrieval of the evidence complaint among 5,629 documents (68 held-out answerable questions). With one relevant document per question, Hit@$k$ equals Recall@$k$ and MAP equals MRR.", "tab:retrieval", "lccccccc",
          ["Retriever", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR", "nDCG@10"],
          [[esc(x["retriever"]), *[num(x[f"hit@{k}"]) for k in (1, 3, 5, 10, 20)], num(x["mrr"]), num(x["ndcg@10"])] for x in r.to_dict("records")], size="footnotesize")
    e = json.loads((d / "qa_extra.json").read_text())
    hits = e["answer_hits"]
    write("qa_hits", "Answer Hit@$k$: the correct answer appears among the top $k$ ranked spans.", "tab:qahits", "lcccc", ["Match rule", "Hit@1", "Hit@3", "Hit@5", "Hit@10"],
          [["Exact match", *[num(hits[f"hit@{k}"]) for k in (1, 3, 5, 10)]], ["Token F1 $\\geq$ 0.5", *[num(hits[f"f1_hit@{k}"]) for k in (1, 3, 5, 10)]]])
    g = e["gate_vs_no_gate"]
    lo, hi = e["overall_accuracy_ci"]
    write("qa_extra", "Additional held-out metrics for the dev-selected system.", "tab:qaextra", "p{9.6cm}c", ["Metric", "Value"], [
        ["Answerability AUROC (score separates answerable from unanswerable)", num(e["answerability_auroc"])],
        ["Answerability average precision", num(e["answerability_ap"])],
        ["Rejection precision / recall / F1 at $\\tau$", f"{e['rejection']['precision']:.3f} / {e['rejection']['recall']:.3f} / {e['rejection']['f1']:.3f}"],
        ["Overall accuracy 95\\% CI with abstention", f"{lo:.3f}--{hi:.3f}"],
        ["Answer drawn from the correct complaint", num(e["right_document_rate"])],
        ["Exact match when the correct complaint is used", num(e["em_given_right_document"])],
        ["Errors from the wrong complaint / from the right complaint", f"{e['errors_wrong_document']} / {e['errors_right_document']}"],
        ["Gate vs no gate: correct only with / only without; McNemar $p$", f"{g['gate_only_correct']} / {g['no_gate_only_correct']}; {g['mcnemar_p']:.3f}"],
    ], size="footnotesize")
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


def extended():
    d = RES / "word2vec" / "intrinsic_by_fold.csv"
    w = pd.read_csv(d).drop(columns="fold").groupby("model", sort=False).mean()
    write("w2v_hits", "Neighbour retrieval on held-out pairs. Synonym Hit@10 and MRR should be high; antonym Hit@10 counts an opposite word intruding among the ten nearest neighbours, so lower is better.", "tab:w2vhits", "lcccc",
          ["Model", "Synonym Hit@10", "Synonym MRR", "Antonym Hit@10", "Negation Hit@10"],
          [[esc(m), num(r.syn_hit10), num(r.syn_mrr), num(r.ant_hit10), num(r.neg_hit10)] for m, r in w.iterrows()], bold_rows=[4], size="footnotesize")
    s = RES / "sentiment"
    e = pd.read_csv(s / "extra_metrics.csv")
    write("sent_extra", "Threshold-free and agreement metrics from the saved out-of-fold probabilities.", "tab:sentextra", "lccccccc",
          ["Model", "ROC-AUC", "PR-AUC (Bad)", "MCC", "Kappa", "Bal.\\ acc.", "Brier", "Log loss"],
          [[esc(r.model), num(r.roc_auc), num(r.pr_auc_bad), num(r.mcc), num(r.kappa), num(r.balanced_accuracy), num(r.brier), num(r.log_loss)] for r in e.itertuples()], bold_rows=[0], size="footnotesize")
    c = pd.read_csv(s / "confusion_all.csv").set_index("model").loc[e.model]
    write("sent_confusion", "Out-of-fold confusion counts (4,972 Bad and 4,902 Good reviews).", "tab:sentconfusion", "lcccc",
          ["Model", "Bad as Bad", "Bad as Good", "Good as Bad", "Good as Good"],
          [[esc(m), f"{r.bad_as_bad:,}", f"{r.bad_as_good:,}", f"{r.good_as_bad:,}", f"{r.good_as_good:,}"] for m, r in c.iterrows()], bold_rows=[0], size="footnotesize")
    q = pd.read_csv(s / "triage_queue_hits.csv").set_index("model").loc[e.model]
    write("sent_queue", "Triage queue hit rate: reviews are ranked by predicted negativity and the share of the top $k$ that are truly Bad is reported.", "tab:sentqueue", "lccccc",
          ["Model", "Hit@100", "Hit@500", "Hit@1000", "Hit@2000", "Bad recall@2000"],
          [[esc(m), num(r["bad_hit@100"]), num(r["bad_hit@500"]), num(r["bad_hit@1000"]), num(r["bad_hit@2000"]), num(r["bad_recall@2000"])] for m, r in q.iterrows()], bold_rows=[0], size="footnotesize")
    f = pd.read_csv(RES / "summarization" / "fewsum_scores.csv")
    write("sum_detail", "FewSum test: ROUGE precision and recall, word-distribution distance from the source, and sentiment agreement with the star ratings.", "tab:fewsumdetail", "lcccccc",
          ["Method", "R-1 P", "R-1 R", "R-2 P", "R-2 R", "Unigram JS", "Sentiment hit"],
          [[esc(r.method), num(r.rouge1_p), num(r.rouge1_r), num(r.rouge2_p), num(r.rouge2_r), num(r.unigram_js), num(r.sentiment_hit)] for r in f.itertuples()], size="footnotesize")
    t = pd.read_csv(RES / "summarization" / "telecom_scores.csv")
    write("sum_hits", "Telecom clusters: sentiment hit (summary and cluster agree on the majority sentiment), unigram JS divergence from the source, and compression ratio.", "tab:telecomhits", "lccc",
          ["Method", "Sentiment hit", "Unigram JS", "Compression"],
          [[esc(r.method), num(r.sentiment_hit), num(r.unigram_js), num(r.compression, 4)] for r in t.itertuples()], bold_rows=[4], size="footnotesize")


def integrated():
    demo = json.loads((RES / "integrated" / "demo.json").read_text(encoding="utf-8"))
    write("demo_triage", "Integrated system: live triage output.", "tab:demotriage", "p{8.6cm}ccc", ["Incoming review", "Prediction", "P(Good)", "Decision"],
          [[esc(t["review"]), t["prediction"], num(t["probability_good"]), esc(t["decision"].replace("_", " "))] for t in demo["triage"]], size="footnotesize")
    write("demo_qa", "Integrated system: live question answering with evidence or abstention.", "tab:demoqa", "p{8.0cm}p{2.6cm}cc", ["Question", "Answer", "Score", "Complaint"],
          [[esc(a["question"]), esc(a["answer"] or "(abstained)"), num(a["score"], 2), a.get("doc_id", "--")] for a in demo["answers"]], size="footnotesize")
    (OUT / "demo_digest.tex").write_text(r"\textit{" + esc(" | ".join(demo["digest"]["summary"])) + "}", encoding="utf-8")


if __name__ == "__main__":
    for step in (word2vec, sentiment, summarization, qa, extended, integrated):
        step()
    print("tables written", len(list(OUT.glob("*.tex"))))
