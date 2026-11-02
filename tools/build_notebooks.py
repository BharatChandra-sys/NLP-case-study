import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = """import glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PACKAGES = {packages}
KAGGLE = os.path.isdir("/kaggle/working")
if KAGGLE:
    if PACKAGES:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
    hits = glob.glob("/kaggle/input/**/telelens/__init__.py", recursive=True)
    if hits:
        ROOT = Path(hits[0]).parents[2]
    else:
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/BharatChandra-sys/NLP-case-study.git", "/tmp/repo"], check=True)
        ROOT = Path("/tmp/repo")
    RESULTS = Path("/kaggle/working/results/{topic}")
else:
    ROOT = Path.cwd().resolve().parent
    RESULTS = ROOT / "results" / "{topic}"
sys.path.insert(0, str(ROOT / "src"))
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
print("device", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")


def save_json(name, payload):
    (RESULTS / name).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
"""

W2V = [
("md", "# Case Study 1: Word2Vec\n\nSkip-gram with negative sampling, implemented from scratch, compared with gensim Word2Vec, gensim FastText, and **Polarity-Aware Word2Vec (PA-W2V)**: negation binding plus fold-nested polarity counter-fitting."),
("code", SETUP.format(packages='["gensim"]', topic="word2vec")),
("code", """from gensim.models import FastText, Word2Vec
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from telelens.data import load_simlex, load_telecom
from telelens.embeddings import Embeddings, log_odds_polarity, polarity_counterfit, train_sgns
from telelens.sentiment import dense_lr, grouped_folds
from telelens.text import bind_negation, tokenize
from telelens import viz

frame = load_telecom(10000)
plain = [tokenize(t) for t in frame["text"]]
bound = [bind_negation(d) for d in plain]
labels = frame["label"].to_numpy()
folds = grouped_folds(frame)
print(len(frame), "reviews", sum(map(len, plain)), "tokens", len({t for d in plain for t in d}), "types")
print("negation bound tokens", sum(t.startswith("not_") for d in bound for t in d))"""),
("code", """SYNONYMS = [("broadband", "internet"), ("engineer", "technician"), ("bill", "invoice"), ("rude", "unhelpful"),
            ("quick", "fast"), ("helpful", "friendly"), ("terrible", "awful"), ("cancel", "terminate"),
            ("cheap", "affordable"), ("router", "modem"), ("happy", "pleased"), ("problem", "issue"),
            ("refund", "reimbursement"), ("poor", "bad"), ("excellent", "great"), ("advisor", "agent")]
ANTONYMS = [("good", "bad"), ("fast", "slow"), ("helpful", "unhelpful"), ("reliable", "unreliable"),
            ("stable", "unstable"), ("cheap", "expensive"), ("polite", "rude"), ("happy", "unhappy"),
            ("excellent", "terrible"), ("easy", "difficult"), ("love", "hate"), ("best", "worst"),
            ("great", "poor"), ("friendly", "rude"), ("satisfied", "dissatisfied"), ("efficient", "useless")]
NEGATED = [("good", "not_good"), ("reliable", "not_reliable"), ("helpful", "not_helpful"), ("recommend", "not_recommend"),
           ("working", "not_working"), ("happy", "not_happy"), ("fast", "not_fast"), ("worth", "not_worth"),
           ("resolved", "not_resolved"), ("fixed", "not_fixed")]
HELD_OUT = frozenset(SYNONYMS + ANTONYMS + NEGATED)
simlex = load_simlex()


def pair_hits(emb, pairs, k=10):
    ranks = []
    for a, b in pairs:
        if a in emb and b in emb:
            ordered = [w for w, _ in emb.neighbours(a, len(emb.words))]
            ranks.append(ordered.index(b) + 1)
    if not ranks:
        return np.nan, np.nan
    return float(np.mean([r <= k for r in ranks])), float(np.mean([1 / r for r in ranks]))


def pair_metrics(emb):
    syn_hit, syn_mrr = pair_hits(emb, SYNONYMS)
    ant_hit, ant_mrr = pair_hits(emb, ANTONYMS)
    neg_hit, _ = pair_hits(emb, NEGATED)
    syn = [emb.cosine(a, b) for a, b in SYNONYMS]
    ant = [emb.cosine(a, b) for a, b in ANTONYMS]
    neg = [emb.cosine(a, b) for a, b in NEGATED]
    syn, ant, neg = [[x for x in values if x is not None] for values in (syn, ant, neg)]
    rows = [(s, 1) for s in syn] + [(a, 0) for a in ant]
    sims = [(emb.cosine(a, b), g) for a, b, g in simlex.itertuples(index=False)]
    sims = [(c, g) for c, g in sims if c is not None]
    return {"syn_hit10": syn_hit, "syn_mrr": syn_mrr, "ant_hit10": ant_hit, "ant_mrr": ant_mrr, "neg_hit10": neg_hit,
            "syn_mean": np.mean(syn), "ant_mean": np.mean(ant), "neg_mean": np.mean(neg) if neg else np.nan,
            "syn_ant_auc": roc_auc_score([r[1] for r in rows], [r[0] for r in rows]),
            "simlex_rho": spearmanr([s[0] for s in sims], [s[1] for s in sims]).statistic, "simlex_pairs": len(sims)}


def neighbour_purity(emb, held_out_polarity, z=1.96, k=10):
    polar = {w: s for w, s in held_out_polarity.items() if abs(s) >= z and w in emb}
    words = list(polar)
    ids = [emb.index[w] for w in words]
    sims = emb.vectors[ids] @ emb.vectors[ids].T
    np.fill_diagonal(sims, -1)
    signs = np.sign([polar[w] for w in words])
    top = np.argsort(-sims, axis=1)[:, :k]
    return float((signs[top] == signs[:, None]).mean())


def gensim_embeddings(model):
    return Embeddings(model.wv.index_to_key, model.wv.vectors.copy())


def train_all(idx):
    tr_plain = [plain[i] for i in idx]
    tr_bound = [bound[i] for i in idx]
    y = labels[idx]
    sgns, loss = train_sgns(tr_plain, device=DEVICE)
    bound_sgns, _ = train_sgns(tr_bound, device=DEVICE, bound=True)
    polarity = log_odds_polarity(tr_bound, y)
    pa, stats = polarity_counterfit(bound_sgns, polarity, excluded_pairs=HELD_OUT, device=DEVICE)
    settings = dict(vector_size=200, window=5, min_count=5, sg=1, negative=10, epochs=5, seed=SEED, workers=4)
    gw2v = gensim_embeddings(Word2Vec(tr_plain, **settings))
    fast = gensim_embeddings(FastText(tr_plain, min_n=3, max_n=6, **settings))
    return {"SGNS (scratch)": sgns, "gensim Word2Vec": gw2v, "FastText": fast, "SGNS + negation": bound_sgns, "PA-W2V": pa}, stats, loss"""),
("md", "## Tune counter-fitting on an inner validation split of fold 0"),
("code", """train_idx = folds[0][0]
inner_fit, inner_val = train_test_split(train_idx, test_size=0.2, stratify=labels[train_idx], random_state=SEED)
base, _ = train_sgns([bound[i] for i in inner_fit], device=DEVICE, bound=True)
inner_polarity = log_odds_polarity([bound[i] for i in inner_fit], labels[inner_fit])
grid = []
for repel_weight, attract_weight, anchor_weight in [(1.0, 0.3, 2.0), (1.0, 0.3, 1.0), (2.0, 0.3, 1.0), (1.0, 0.1, 4.0), (0.5, 0.3, 4.0), (2.0, 0.6, 2.0)]:
    emb, _ = polarity_counterfit(base, inner_polarity, excluded_pairs=HELD_OUT, device=DEVICE, repel_weight=repel_weight,
                                 attract_weight=attract_weight, anchor_weight=anchor_weight)
    x_fit, x_val = emb.embed_many([bound[i] for i in inner_fit]), emb.embed_many([bound[i] for i in inner_val])
    prob = dense_lr(x_fit, labels[inner_fit], x_val)
    grid.append({"repel": repel_weight, "attract": attract_weight, "anchor": anchor_weight,
                 "val_macro_f1": f1_score(labels[inner_val], prob >= 0.5, average="macro"), **pair_metrics(emb)})
x_fit, x_val = base.embed_many([bound[i] for i in inner_fit]), base.embed_many([bound[i] for i in inner_val])
print("no counter-fit val macro-F1", f1_score(labels[inner_val], dense_lr(x_fit, labels[inner_fit], x_val) >= 0.5, average="macro"))
tuning = pd.DataFrame(grid).sort_values("val_macro_f1", ascending=False)
tuning.to_csv(RESULTS / "counterfit_tuning.csv", index=False)
tuning.round(4)"""),
("md", "## Five-fold evaluation with label use nested inside each training fold"),
("code", """rows, stats_rows, losses = [], [], None
start = time.time()
for fold, (tr, te) in enumerate(folds):
    models, stats, loss = train_all(tr)
    losses = losses or loss
    stats_rows.append({"fold": fold, **stats})
    held_plain = log_odds_polarity([plain[i] for i in te], labels[te])
    held_bound = log_odds_polarity([bound[i] for i in te], labels[te])
    for name, emb in models.items():
        heldout = held_bound if emb.bound else held_plain
        rows.append({"fold": fold, "model": name, "vocab": len(emb.words), "neighbour_purity": neighbour_purity(emb, heldout), **pair_metrics(emb)})
    print("fold", fold, round(time.time() - start), "s")
fold_table = pd.DataFrame(rows)
fold_table.to_csv(RESULTS / "intrinsic_by_fold.csv", index=False)
summary = fold_table.drop(columns="fold").groupby("model", sort=False).agg(["mean", "std"]).round(4)
summary.to_csv(RESULTS / "intrinsic_summary.csv")
save_json("sgns_loss.json", losses)
pd.DataFrame(stats_rows).to_csv(RESULTS / "counterfit_constraints.csv", index=False)
summary"""),
("md", "## Whole-corpus model: neighbours, negation probes and PCA"),
("code", """all_idx = np.arange(len(frame))
final, final_stats, _ = train_all(all_idx)
QUERY = ["reliable", "unreliable", "not_reliable", "stable", "helpful", "engineer", "bill", "refund", "fast", "not_working"]
table = []
for word in QUERY:
    for name in ("SGNS + negation", "PA-W2V"):
        table.append({"word": word, "model": name, "neighbours": ", ".join(w for w, _ in final[name].neighbours(word, 6))})
neighbours = pd.DataFrame(table)
neighbours.to_csv(RESULTS / "neighbours.csv", index=False)

PROBES = [("the service is reliable", "the service is not reliable"), ("the connection is stable", "the connection is unstable"),
          ("the engineer was helpful", "the engineer was not helpful"), ("my problem was fixed", "my problem was not fixed"),
          ("broadband is slow and keeps disconnecting", "internet connection is sluggish and drops")]
probes = []
for a, b in PROBES:
    row = {"a": a, "b": b}
    for name, emb in final.items():
        tok = (lambda s: bind_negation(tokenize(s))) if emb.bound else tokenize
        row[name] = float(emb.embed(tok(a)) @ emb.embed(tok(b)))
    probes.append(row)
probes = pd.DataFrame(probes).round(3)
probes.to_csv(RESULTS / "doc_probes.csv", index=False)
display(neighbours.head(8), probes)"""),
("code", """WORDS = ["good", "great", "excellent", "helpful", "friendly", "reliable", "fast", "easy", "happy", "bad", "terrible", "awful", "rude",
         "unhelpful", "unreliable", "slow", "useless", "poor", "not_reliable", "not_helpful", "not_good", "not_working", "unstable", "stable"]
polarity = log_odds_polarity(bound, labels)
points = {}
for name in ("SGNS + negation", "PA-W2V"):
    emb = final[name]
    present = [w for w in WORDS if w in emb]
    xy = PCA(2, random_state=SEED).fit_transform(emb.vectors[[emb.index[w] for w in present]])
    points[name] = [(w, x, y, 1 if polarity.get(w, 0) > 0 else -1) for w, (x, y) in zip(present, xy)]
save_json("pca_points.json", points)
viz.pca_words(RESULTS / "word2vec_pca.png", points, "Selected word vectors, two PCA components")
means = fold_table.groupby("model", sort=False).mean(numeric_only=True)
viz.grouped(RESULTS / "syn_ant_cosine.png", list(means.index), [("Synonyms", means.syn_mean.tolist()), ("Antonyms", means.ant_mean.tolist())],
            "Mean cosine on held-out word pairs", "Cosine")
save_json("final_constraints.json", final_stats)"""),
]

SENT = [
("md", "# Case Study 2: Sentiment Classification\n\nGrouped five-fold comparison, calibrated probabilities, and selective triage that defers uncertain reviews to an analyst."),
("code", SETUP.format(packages='["gensim"]', topic="sentiment")),
("code", """import re
from gensim.models import FastText
from sklearn.metrics import confusion_matrix, f1_score
from telelens.data import load_comcast, load_telecom
from telelens.embeddings import Embeddings, log_odds_polarity, polarity_counterfit, train_sgns
from telelens.metrics import (bootstrap_ci, bootstrap_macro_f1_diff, classification_report, expected_calibration_error,
                              mcnemar_exact, risk_coverage, threshold_for_risk)
from telelens.sentiment import dense_lr, distilbert_finetune, grouped_folds, hybrid_lr, pretrained_sst2, tfidf_lr
from telelens.text import bind_negation, tokenize
from telelens import viz

frame = load_telecom(10000)
small = load_telecom(1000)
y = frame["label"].to_numpy()
texts = frame["text"].tolist()
plain = [tokenize(t) for t in texts]
bound = [bind_negation(d) for d in plain]
folds = grouped_folds(frame)
print(len(frame), np.bincount(y))"""),
("md", "## Continuity with the earlier report: TF-IDF on the 1K file"),
("code", """oof = np.zeros(len(small))
for tr, te in grouped_folds(small):
    oof[te] = tfidf_lr(small["text"].iloc[tr], small["label"].iloc[tr], small["text"].iloc[te])
continuity = classification_report(small["label"], (oof >= 0.5).astype(int))
save_json("continuity_1k.json", continuity)
continuity"""),
("md", "## Five-fold comparison on the 10K file"),
("code", """names = ["TF-IDF LR", "SGNS mean LR", "FastText mean LR", "SGNS+neg mean LR", "PA-W2V mean LR", "TF-IDF + SGNS", "TF-IDF + PA-W2V", "DistilBERT fine-tuned"]
probs = {n: np.zeros(len(frame)) for n in names}
fold_of = np.zeros(len(frame), dtype=int)
temps = []
timing = {n: 0.0 for n in names}


def timed(name, fn):
    start = time.time()
    value = fn()
    timing[name] += time.time() - start
    return value


for fold, (tr, te) in enumerate(folds):
    fold_of[te] = fold
    x_tr, x_te = [texts[i] for i in tr], [texts[i] for i in te]
    probs["TF-IDF LR"][te] = timed("TF-IDF LR", lambda: tfidf_lr(x_tr, y[tr], x_te))
    sgns, _ = train_sgns([plain[i] for i in tr], device=DEVICE)
    a, b = sgns.embed_many([plain[i] for i in tr]), sgns.embed_many([plain[i] for i in te])
    probs["SGNS mean LR"][te] = timed("SGNS mean LR", lambda: dense_lr(a, y[tr], b))
    probs["TF-IDF + SGNS"][te] = hybrid_lr(x_tr, a, y[tr], x_te, b)
    fast = FastText([plain[i] for i in tr], vector_size=200, window=5, min_count=5, sg=1, negative=10, epochs=5, seed=SEED, workers=4, min_n=3, max_n=6)
    emb = Embeddings(fast.wv.index_to_key, fast.wv.vectors.copy())
    probs["FastText mean LR"][te] = dense_lr(emb.embed_many([plain[i] for i in tr]), y[tr], emb.embed_many([plain[i] for i in te]))
    base, _ = train_sgns([bound[i] for i in tr], device=DEVICE, bound=True)
    a, b = base.embed_many([bound[i] for i in tr]), base.embed_many([bound[i] for i in te])
    probs["SGNS+neg mean LR"][te] = dense_lr(a, y[tr], b)
    pa, _ = polarity_counterfit(base, log_odds_polarity([bound[i] for i in tr], y[tr]), device=DEVICE)
    a, b = pa.embed_many([bound[i] for i in tr]), pa.embed_many([bound[i] for i in te])
    probs["PA-W2V mean LR"][te] = dense_lr(a, y[tr], b)
    probs["TF-IDF + PA-W2V"][te] = hybrid_lr(x_tr, a, y[tr], x_te, b)
    p, temp = timed("DistilBERT fine-tuned", lambda: distilbert_finetune(x_tr, y[tr], x_te, DEVICE))
    probs["DistilBERT fine-tuned"][te] = p
    temps.append(temp)
    print("fold", fold, {k: round(f1_score(y[te], probs[k][te] >= 0.5, average="macro"), 4) for k in names})
probs["DistilBERT SST-2 (no training)"] = pretrained_sst2(texts, DEVICE)
pd.DataFrame({"label": y, "fold": fold_of, **probs}).to_csv(RESULTS / "oof_probabilities.csv", index_label="row")"""),
("code", """rows = []
for name, p in probs.items():
    pred = (p >= 0.5).astype(int)
    fold_f1 = [f1_score(y[te], pred[te], average="macro") for _, te in folds]
    lo, hi = bootstrap_ci((pred == y).astype(float))
    rows.append({"model": name, **classification_report(y, pred), "fold_f1_std": np.std(fold_f1), "acc_lo": lo, "acc_hi": hi,
                 "ece": expected_calibration_error(y, p)})
scores = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
scores.to_csv(RESULTS / "scores.csv", index=False)
comparisons = []
for a, b in [("TF-IDF + PA-W2V", "TF-IDF LR"), ("TF-IDF + PA-W2V", "TF-IDF + SGNS"), ("PA-W2V mean LR", "SGNS mean LR"), ("PA-W2V mean LR", "SGNS+neg mean LR"),
             ("SGNS+neg mean LR", "SGNS mean LR"), ("DistilBERT fine-tuned", "TF-IDF LR"), ("DistilBERT fine-tuned", "TF-IDF + PA-W2V")]:
    pa, pb = (probs[a] >= 0.5).astype(int), (probs[b] >= 0.5).astype(int)
    mean, lo, hi = bootstrap_macro_f1_diff(y, pa, pb)
    a_only, b_only, p_value = mcnemar_exact(y, pa, pb)
    comparisons.append({"a": a, "b": b, "f1_diff": mean, "ci_lo": lo, "ci_hi": hi, "a_only_correct": a_only, "b_only_correct": b_only, "mcnemar_p": p_value})
comparisons = pd.DataFrame(comparisons)
comparisons.to_csv(RESULTS / "comparisons.csv", index=False)
save_json("timing_seconds.json", timing)
save_json("temperatures.json", temps)
display(scores.round(4), comparisons.round(4))"""),
("md", "## Selective triage: deferral thresholds are cross-fitted across folds"),
("code", """TARGETS = [0.10, 0.05, 0.03, 0.02]
selective = []
for name in ["TF-IDF LR", "TF-IDF + PA-W2V", "DistilBERT fine-tuned"]:
    p = probs[name]
    for target in TARGETS:
        kept = np.zeros(len(y), dtype=bool)
        for k in range(5):
            tau = threshold_for_risk(y[fold_of != k], p[fold_of != k], target)
            kept[fold_of == k] = np.maximum(p[fold_of == k], 1 - p[fold_of == k]) >= tau
        pred = (p >= 0.5).astype(int)
        bad_auto = (y == 0) & kept
        selective.append({"model": name, "target_risk": target, "coverage": kept.mean(), "risk": 1 - (pred[kept] == y[kept]).mean(),
                          "deferred": int((~kept).sum()), "auto_bad_recall": (pred[bad_auto] == 0).mean(),
                          "missed_bad_auto": int(((pred == 1) & bad_auto).sum())})
selective = pd.DataFrame(selective)
selective.to_csv(RESULTS / "selective.csv", index=False)
curves = []
for name in ["TF-IDF LR", "TF-IDF + PA-W2V", "DistilBERT fine-tuned"]:
    rc = risk_coverage(y, probs[name])
    curves.append((name, [r["coverage"] for r in rc], [r["risk"] for r in rc]))
viz.lines(RESULTS / "risk_coverage.png", curves, "Risk falls as more reviews are deferred", "Coverage (share handled automatically)", "Error rate on handled reviews", xlim=(0.2, 1.0))
ordered = scores.sort_values("macro_f1", ascending=False)
viz.hbar(RESULTS / "model_f1.png", ordered["model"].tolist(), ordered["macro_f1"].tolist(), "Out-of-fold macro-F1, 10K grouped CV", "Macro-F1", xlim=(0.6, 1.0))
best = (probs["DistilBERT fine-tuned"] >= 0.5).astype(int)
save_json("confusion_distilbert.json", confusion_matrix(y, best).tolist())
selective.round(4)"""),
("md", "## Robustness: contrast set and cross-domain test on Comcast complaints"),
("code", """CONTRAST = [("the service is reliable", 1), ("the service is not reliable", 0), ("the connection is stable", 1), ("the connection is unstable", 0),
            ("the engineer was helpful", 1), ("the engineer was not helpful", 0), ("my problem was fixed quickly", 1), ("my problem was never fixed", 0),
            ("i would recommend this provider", 1), ("i would not recommend this provider", 0), ("the staff were polite", 1), ("the staff were not polite", 0),
            ("the broadband is fast", 1), ("the broadband is not fast", 0), ("billing is clear and easy", 1), ("billing is not clear or easy", 0),
            ("no problems at all with the router", 1), ("nothing but problems with the router", 0), ("i am happy with the price", 1), ("i am not happy with the price", 0),
            ("she was very friendly", 1), ("he was very unfriendly", 0), ("installation was smooth", 1), ("installation was a nightmare", 0),
            ("never had an issue with the signal", 1), ("always have an issue with the signal", 0), ("not bad at all", 1), ("not good at all", 0),
            ("customer support could not be better", 1), ("customer support could not be worse", 0)]
contrast_x, contrast_y = [c[0] for c in CONTRAST], np.array([c[1] for c in CONTRAST])
comcast = load_comcast()
comcast = comcast[comcast["rating"].isin([1, 4, 5])]
clean = [" ".join(re.findall(r"[a-z]+", t.lower().replace("'", ""))) for t in comcast["text"]]
comcast_y = (comcast["rating"] >= 4).astype(int).to_numpy()
base, _ = train_sgns(bound, device=DEVICE, bound=True)
pa, _ = polarity_counterfit(base, log_odds_polarity(bound, y), device=DEVICE)
train_x = pa.embed_many(bound)
embed = lambda items: pa.embed_many([bind_negation(tokenize(t)) for t in items])
targets = contrast_x + clean
outputs = {"TF-IDF LR": tfidf_lr(texts, y, targets), "TF-IDF + PA-W2V": hybrid_lr(texts, train_x, y, targets, embed(targets)),
           "PA-W2V mean LR": dense_lr(train_x, y, embed(targets)), "DistilBERT fine-tuned": distilbert_finetune(texts, y, targets, DEVICE)[0]}
n = len(contrast_x)
robust = []
contrast_detail = pd.DataFrame({"text": contrast_x, "label": contrast_y})
for name, p in outputs.items():
    contrast_detail[name] = p[:n].round(3)
    robust.append({"model": name, "contrast_accuracy": ((p[:n] >= 0.5) == contrast_y).mean(), "comcast_macro_f1": f1_score(comcast_y, p[n:] >= 0.5, average="macro"),
                   **{f"comcast_{k}": v for k, v in classification_report(comcast_y, (p[n:] >= 0.5).astype(int)).items() if k in ("accuracy", "bad_recall")},
                   "comcast_good_n": int(comcast_y.sum()), "comcast_bad_n": int((1 - comcast_y).sum())})
robust = pd.DataFrame(robust)
robust.to_csv(RESULTS / "robustness.csv", index=False)
contrast_detail.to_csv(RESULTS / "contrast_detail.csv", index=False)
robust.round(4)"""),
]

SUMM = [
("md", "# Case Study 3: Text Summarization\n\n**PAB-MMR**: Polarity- and Aspect-Balanced MMR. Evaluated against human references on FewSum and on telecom digests where no references exist."),
("code", SETUP.format(packages='["rouge-score", "sentencepiece", "protobuf"]', topic="summarization")),
("code", """import itertools
from scipy.stats import wilcoxon
from telelens import summarize as S
from telelens.data import load_fewsum, load_telecom
from telelens.embeddings import log_odds_polarity, polarity_counterfit, train_sgns
from telelens.metrics import bootstrap_ci, js_binary
from telelens.punctuate import PunctuationRestorer
from telelens.sentiment import grouped_folds, make_vectorizer, pretrained_sst2
from telelens.text import bind_negation, boundary_prf, segment_fixed, segment_unpunctuated, split_sentences, tokenize
from telelens import viz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import re

dev_groups = load_fewsum("train") + load_fewsum("val")
test_groups = load_fewsum("test")
BUDGET = int(np.mean([S.word_count(r) for g in dev_groups for r in g["references"]]))
print(len(dev_groups), len(test_groups), "budget", BUDGET)"""),
("md", "## FewSum: reviews, human references and a dev/test split"),
("code", """all_sentences = {}
def prepare(group):
    sentences = S.filter_sentences([s for r in group["reviews"] for s in split_sentences(r)])
    vectors = S.sentence_vectors(sentences)
    return {"sentences": sentences, "vectors": vectors, "aspects": S.kmeans_aspects(vectors), "references": group["references"],
            "reviews": group["reviews"], "target": float(np.mean([(r - 1) / 4 for r in group["ratings"]]))}
prepared_dev = [prepare(g) for g in dev_groups]
prepared_test = [prepare(g) for g in test_groups]
flat = [s for g in prepared_dev + prepared_test for s in g["sentences"]]
scores = pretrained_sst2(flat, DEVICE)
cursor = 0
for g in prepared_dev + prepared_test:
    g["polarity"] = scores[cursor : cursor + len(g["sentences"])]
    cursor += len(g["sentences"])


def run_pab(g, lam, beta, eta):
    return S.pab_mmr(g["vectors"], g["sentences"], BUDGET, polarity=g["polarity"], aspects=g["aspects"], target=g["target"], lam=lam, beta=beta, eta=eta)


def text_of(g, chosen):
    return " ".join(g["sentences"][i] for i in chosen)


grid = []
for lam, beta, eta in itertools.product([0.3, 0.5, 0.7, 0.9], [0.0, 0.1, 0.3], [0.0, 0.25, 0.5, 1.0]):
    r2 = np.mean([S.rouge_multi(text_of(g, run_pab(g, lam, beta, eta)), g["references"])["rouge2"] for g in prepared_dev])
    grid.append({"lam": lam, "beta": beta, "eta": eta, "dev_rouge2": r2})
grid = pd.DataFrame(grid).sort_values("dev_rouge2", ascending=False)
grid.to_csv(RESULTS / "fewsum_dev_grid.csv", index=False)
best_pab = grid.iloc[0]
best_mmr = grid[(grid.beta == 0) & (grid.eta == 0)].iloc[0]
FIXED = dict(lam=float(best_mmr.lam), beta=0.3, eta=0.5)
print(best_pab.to_dict(), best_mmr.to_dict(), FIXED)
grid.head(6).round(4)"""),
("code", """documents = [" ".join(g["reviews"]) for g in prepared_test]
bart = S.bart_summaries(documents, DEVICE)
def fewsum_outputs(g, i):
    v, s = g["vectors"], g["sentences"]
    return {"Lead": text_of(g, S.lead(s, BUDGET)), "Centroid": text_of(g, S.centroid(v, s, BUDGET)), "TextRank": text_of(g, S.textrank(v, s, BUDGET)),
            "MMR (tuned)": text_of(g, run_pab(g, best_mmr.lam, 0.0, 0.0)), "PAB-MMR (tuned)": text_of(g, run_pab(g, best_pab.lam, best_pab.beta, best_pab.eta)),
            "PAB-MMR (fixed)": text_of(g, run_pab(g, **FIXED)), "BART-large-CNN": bart[i]}
rows, gaps, samples, texts = [], [], [], []
for i, g in enumerate(prepared_test):
    outputs = fewsum_outputs(g, i)
    for name, summary in outputs.items():
        texts.append(summary)
        rows.append({"group": i, "method": name, "words": S.word_count(summary), **S.rouge_multi(summary, g["references"]),
                     "unigram_js": S.unigram_js(summary, " ".join(g["reviews"])), "target": g["target"]})
    samples.append({"group": test_groups[i]["group"], **outputs, "reference": g["references"][0]})
summary_polarity = pretrained_sst2(texts, DEVICE)
for row, value in zip(rows, summary_polarity):
    row["polarity_gap"] = js_binary(float(value), row["target"])
    row["sentiment_hit"] = float((value >= 0.5) == (row["target"] >= 0.5))
    del row["target"]
fewsum = pd.DataFrame(rows)
fewsum.to_csv(RESULTS / "fewsum_by_group.csv", index=False)
agg = fewsum.drop(columns="group").groupby("method", sort=False).mean()
agg["rouge2_lo"], agg["rouge2_hi"] = zip(*[bootstrap_ci(fewsum[fewsum.method == m].rouge2) for m in agg.index])
agg.to_csv(RESULTS / "fewsum_scores.csv")
tests = []
for a, b in [("PAB-MMR (tuned)", "MMR (tuned)"), ("PAB-MMR (fixed)", "MMR (tuned)"), ("PAB-MMR (tuned)", "TextRank"), ("PAB-MMR (tuned)", "BART-large-CNN")]:
    x, y = fewsum[fewsum.method == a].rouge2.to_numpy(), fewsum[fewsum.method == b].rouge2.to_numpy()
    tests.append({"a": a, "b": b, "mean_diff": (x - y).mean(), "wilcoxon_p": wilcoxon(x, y).pvalue if np.any(x != y) else 1.0})
pd.DataFrame(tests).to_csv(RESULTS / "fewsum_tests.csv", index=False)
pd.DataFrame(samples).to_csv(RESULTS / "fewsum_samples.csv", index=False)
display(agg.round(4), pd.DataFrame(tests).round(4))"""),
("md", "## Segmenting text that has no punctuation: measured by stripping FewSum punctuation"),
("code", """strip = lambda s: " ".join(re.sub(r"[^a-z\\s]", " ", s.lower()).split())
restorer = PunctuationRestorer(DEVICE)
truths, joined = [], []
for g in prepared_dev + prepared_test:
    for review in g["reviews"]:
        truth = [t for t in (strip(x) for x in split_sentences(review)) if t]
        if len(truth) > 1:
            truths.append(truth)
            joined.append(" ".join(truth))
restored = restorer.segment_many(joined)
candidates = {"Cue words": [segment_unpunctuated(t) for t in joined], "Fixed 15-word windows": [segment_fixed(t) for t in joined], "Punctuation restoration": restored}
segmentation = pd.DataFrame([{"method": m, **dict(zip(["precision", "recall", "f1"], np.mean([boundary_prf(t, p) for t, p in zip(truths, parts)], axis=0))), "reviews": len(truths)} for m, parts in candidates.items()])
stripped = {}
for method in candidates:
    values = []
    for g in prepared_test:
        cleaned = [strip(r) for r in g["reviews"]]
        if method == "Cue words":
            pieces = [segment_unpunctuated(c) for c in cleaned]
        elif method == "Fixed 15-word windows":
            pieces = [segment_fixed(c) for c in cleaned]
        else:
            pieces = restorer.segment_many(cleaned)
        sentences = S.filter_sentences([x for part in pieces for x in part])
        chosen = S.pab_mmr(S.sentence_vectors(sentences), sentences, BUDGET, lam=float(best_mmr.lam))
        values.append(S.rouge_multi(" ".join(sentences[i] for i in chosen), [strip(r) for r in g["references"]])["rouge2"])
    stripped[method] = float(np.mean(values))
segmentation.to_csv(RESULTS / "segmentation.csv", index=False)
save_json("stripped_rouge2.json", stripped)
display(segmentation.round(4), stripped)"""),
("md", "## Telecom digests: provider and provider-by-aspect clusters, with no references available"),
("code", """frame = load_telecom(10000)
frame = frame.reset_index(drop=True)
plain = [tokenize(t) for t in frame["text"]]
bound = [bind_negation(d) for d in plain]
base, _ = train_sgns(bound, device=DEVICE, bound=True)
pa, _ = polarity_counterfit(base, log_odds_polarity(bound, frame["label"]), device=DEVICE)
lexicon = S.expand_aspects(pa)
save_json("aspect_lexicon.json", {k: sorted(v) for k, v in lexicon.items()})
review_vectorizers = []
segment_rows = []
for doc, pieces in enumerate(restorer.segment_many(frame["text"].tolist())):
    for seg in S.filter_sentences(pieces, min_words=5):
        segment_rows.append((doc, seg))
segments = pd.DataFrame(segment_rows, columns=["doc", "text"])
segments["polarity"] = 0.0
for tr, te in grouped_folds(frame):
    vectorizer = make_vectorizer()
    model = CalibratedClassifierCV(LogisticRegression(C=4.0, max_iter=2000), method="sigmoid", cv=3).fit(vectorizer.fit_transform(frame["text"].iloc[tr]), frame["label"].iloc[tr])
    mask = segments["doc"].isin(set(te))
    segments.loc[mask, "polarity"] = model.predict_proba(vectorizer.transform(segments.loc[mask, "text"]))[:, 1]
segments["aspect"] = S.lexicon_aspects(segments["text"].tolist(), lexicon)
global_vectorizer = make_vectorizer()
global_model = CalibratedClassifierCV(LogisticRegression(C=4.0, max_iter=2000), method="sigmoid", cv=3).fit(global_vectorizer.fit_transform(frame["text"]), frame["label"])
PROVIDERS = ["virgin", "bt", "talktalk", "vodafone", "ee", "sky", "tesco", "zen"]
rng = np.random.default_rng(SEED)
clusters = []
for provider in PROVIDERS:
    docs = frame.index[[provider in t.split() for t in frame["text"]]].to_numpy()
    clusters.append((provider, "all", docs))
    for aspect in S.ASPECT_SEEDS:
        chosen = segments[(segments.doc.isin(docs)) & (segments.aspect == aspect)].doc.unique()
        if len(chosen) >= 30:
            clusters.append((provider, aspect, chosen))
print(len(clusters), "clusters")"""),
("code", """rows, examples = [], []
bart_inputs, cluster_payload = [], []
for provider, aspect, docs in clusters:
    docs = rng.choice(docs, size=min(60, len(docs)), replace=False)
    part = segments[segments.doc.isin(set(docs))]
    if aspect != "all":
        part = part[part.aspect == aspect]
    sentences = part["text"].tolist()
    vectors = S.sentence_vectors(sentences)
    target = float(frame.loc[docs, "label"].mean())
    cluster_payload.append((provider, aspect, docs, sentences, vectors, part["polarity"].to_numpy(), part["aspect"].to_numpy(), S.kmeans_aspects(vectors, 4), target))
    bart_inputs.append(" ".join(frame.loc[docs, "text"].tolist())[:6000])
bart_telecom = S.bart_summaries(bart_inputs, DEVICE, max_length=70, min_length=30)
for c, (provider, aspect, docs, sentences, vectors, polarity, aspect_labels, sub, target) in enumerate(cluster_payload):
    facets = aspect_labels if aspect == "all" else sub
    centre = vectors.mean(0)
    picks = {"Lead": S.lead(sentences, BUDGET), "Centroid": S.centroid(vectors, sentences, BUDGET), "TextRank": S.textrank(vectors, sentences, BUDGET),
             "MMR": S.pab_mmr(vectors, sentences, BUDGET, lam=float(best_mmr.lam)),
             "PAB-MMR": S.pab_mmr(vectors, sentences, BUDGET, polarity=polarity, aspects=facets, target=target, **FIXED)}
    for name, chosen in picks.items():
        rows.append({"cluster": c, "provider": provider, "aspect": aspect, "method": name, "target_share_good": target,
                     "polarity_gap": S.polarity_gap(polarity, sentences, chosen, target), "aspect_recall": S.aspect_recall(facets, chosen),
                     "redundancy": S.redundancy(vectors, chosen), "words": sum(S.word_count(sentences[i]) for i in chosen),
                     "centroid_cosine": float(np.dot(vectors[chosen].mean(0), centre) / (np.linalg.norm(vectors[chosen].mean(0)) * np.linalg.norm(centre) + 1e-9)),
                     "sentiment_hit": float((S.weighted_polarity(polarity, sentences, chosen) >= 0.5) == (target >= 0.5)),
                     "unigram_js": S.unigram_js(" ".join(sentences[i] for i in chosen), " ".join(sentences)),
                     "compression": sum(S.word_count(sentences[i]) for i in chosen) / sum(map(S.word_count, sentences))})
    summary = bart_telecom[c]
    bart_segments = S.filter_sentences(split_sentences(summary)) or [summary]
    bart_polarity = global_model.predict_proba(global_vectorizer.transform(bart_segments))[:, 1]
    rows.append({"cluster": c, "provider": provider, "aspect": aspect, "method": "BART-large-CNN", "target_share_good": target, "polarity_gap": js_binary(float(bart_polarity.mean()), target),
                 "aspect_recall": np.nan, "redundancy": np.nan, "words": S.word_count(summary), "centroid_cosine": np.nan, "novel_token_rate": S.novel_token_rate(summary, bart_inputs[c]),
                 "sentiment_hit": float((bart_polarity.mean() >= 0.5) == (target >= 0.5)), "unigram_js": S.unigram_js(summary, bart_inputs[c]),
                 "compression": S.word_count(summary) / S.word_count(bart_inputs[c])})
    if aspect == "all":
        examples.append({"provider": provider, "share_good": target, "PAB-MMR": " | ".join(sentences[i] for i in picks["PAB-MMR"]), "MMR": " | ".join(sentences[i] for i in picks["MMR"]), "BART": summary})
telecom = pd.DataFrame(rows)
telecom.to_csv(RESULTS / "telecom_by_cluster.csv", index=False)
telecom_agg = telecom.groupby("method", sort=False)[["polarity_gap", "aspect_recall", "redundancy", "centroid_cosine", "words", "novel_token_rate", "sentiment_hit", "unigram_js", "compression"]].mean()
telecom_agg.to_csv(RESULTS / "telecom_scores.csv")
pd.DataFrame(examples).to_csv(RESULTS / "telecom_digests.csv", index=False)
tests = []
for metric, base_name in [("polarity_gap", "MMR"), ("aspect_recall", "MMR"), ("redundancy", "MMR"), ("polarity_gap", "TextRank"), ("polarity_gap", "BART-large-CNN")]:
    a = telecom[telecom.method == "PAB-MMR"].set_index("cluster")[metric]
    b = telecom[telecom.method == base_name].set_index("cluster")[metric]
    tests.append({"metric": metric, "versus": base_name, "pab_mean": a.mean(), "other_mean": b.mean(), "wilcoxon_p": wilcoxon(a, b).pvalue, "clusters": len(a)})
pd.DataFrame(tests).to_csv(RESULTS / "telecom_tests.csv", index=False)
gap = telecom_agg["polarity_gap"]
viz.hbar(RESULTS / "telecom_polarity_gap.png", gap.index.tolist(), gap.tolist(), "Lower is better: sentiment drift from the review cluster", "Jensen-Shannon divergence", highlight="PAB-MMR")
r2 = agg["rouge2"].sort_values(ascending=False)
viz.hbar(RESULTS / "fewsum_rouge2.png", r2.index.tolist(), r2.tolist(), "ROUGE-2 F1 against human references, FewSum test", "ROUGE-2 F1", highlight="PAB-MMR (tuned)")
display(telecom_agg.round(4), pd.DataFrame(tests).round(4), pd.DataFrame(examples).head(3))"""),
]

QA = [
("md", "# Case Study 4: Factoid Question Answering\n\nRetriever-reader pipeline over Comcast complaints with hybrid retrieval, answer-type gating and calibrated abstention. Readers are also benchmarked on SQuAD."),
("code", SETUP.format(packages='[]', topic="qa")),
("code", """from telelens.data import load_comcast, load_gold, load_squad, load_telecom
from telelens.embeddings import log_odds_polarity, polarity_counterfit, train_sgns
from telelens.metrics import bootstrap_ci, exact_match, retrieval_scores, selective_qa, token_f1
from telelens.qa import HybridRetriever, QAPipeline, Reader, answer_type, make_passages, rule_answer
from telelens.text import bind_negation, tokenize
from telelens import viz

comcast = load_comcast()
telecom = load_telecom(10000)
gold = load_gold(ROOT / "data" / "gold" / "qa_gold.jsonl")
dev, test = [g for g in gold if g["split"] == "dev"], [g for g in gold if g["split"] == "test"]
passages = make_passages(comcast)
print(len(comcast), "documents", len(passages), "passages", len(dev), "dev", len(test), "test",
      sum(g["answerable"] for g in test), "answerable test")"""),
("md", "## Reader benchmark on SQuAD with the gold paragraph supplied"),
("code", """squad = load_squad()
rng = np.random.default_rng(SEED)
sample = [squad[i] for i in rng.choice(len(squad), 2000, replace=False)]
READERS = {"DistilBERT-SQuAD": "distilbert-base-cased-distilled-squad", "RoBERTa-SQuAD2": "deepset/roberta-base-squad2"}
readers = {name: Reader(model, DEVICE) for name, model in READERS.items()}
bench = []
for name, reader in readers.items():
    start = time.time()
    spans = reader.spans([s["question"] for s in sample], [s["context"] for s in sample])
    elapsed = time.time() - start
    em = [exact_match(found[0]["text"] if found else "", s["answers"]) for found, s in zip(spans, sample)]
    f1 = [token_f1(found[0]["text"] if found else "", s["answers"]) for found, s in zip(spans, sample)]
    lo, hi = bootstrap_ci(em)
    bench.append({"reader": name, "questions": len(sample), "em": np.mean(em), "em_lo": lo, "em_hi": hi, "f1": np.mean(f1), "seconds": elapsed})
bench = pd.DataFrame(bench)
bench.to_csv(RESULTS / "squad_reader_benchmark.csv", index=False)
bench.round(4)"""),
("md", "## Embeddings trained on both telecom corpora; gold questions are never used"),
("code", """texts = comcast["text"].tolist() + telecom["text"].tolist()
ratings = comcast["rating"].tolist()
labels = [1 if r >= 4 else 0 if r == 1 else -1 for r in ratings] + telecom["label"].tolist()
plain = [tokenize(t) for t in texts]
bound = [bind_negation(d) for d in plain]
sgns, _ = train_sgns(plain, device=DEVICE)
base, _ = train_sgns(bound, device=DEVICE, bound=True)
known = [i for i, y in enumerate(labels) if y >= 0]
pa, stats = polarity_counterfit(base, log_odds_polarity([bound[i] for i in known], [labels[i] for i in known]), device=DEVICE)
retriever = HybridRetriever(passages, {"sgns": sgns, "pa_w2v": pa})
print(stats)"""),
("md", "## Retrieval: document rank of the evidence complaint"),
("code", """MODES = {"BM25": "bm25", "SGNS dense": "sgns", "PA-W2V dense": "pa_w2v", "BM25 + SGNS": "bm25+sgns", "BM25 + PA-W2V": "bm25+pa_w2v",
         "BM25 rerank SGNS 0.25": "rerank:sgns:0.25", "BM25 rerank SGNS 0.5": "rerank:sgns:0.5",
         "BM25 rerank PA-W2V 0.25": "rerank:pa_w2v:0.25", "BM25 rerank PA-W2V 0.5": "rerank:pa_w2v:0.5"}
retrieval = []
for label, mode in MODES.items():
    for split, items in (("dev", dev), ("test", test)):
        ranks = [retriever.doc_rank(retriever.rank(g["question"], mode, 100), g["doc_id"]) for g in items if g["answerable"]]
        retrieval.append({"retriever": label, "split": split, **retrieval_scores(ranks)})
retrieval = pd.DataFrame(retrieval)
retrieval.to_csv(RESULTS / "retrieval.csv", index=False)
best_mode_label = retrieval[retrieval.split == "dev"].sort_values("mrr", ascending=False).iloc[0]["retriever"]
print("best on dev", best_mode_label)
retrieval.round(4)"""),
("md", "## Tune reader, gate, retrieval weight and abstention threshold on dev only"),
("code", """pipelines = {(label, name): QAPipeline(retriever, reader, mode=MODES[label]) for label in MODES for name, reader in readers.items()}


def evaluate(pipeline, items, gate, weight):
    results = []
    for g in items:
        best, _ = pipeline.answer(g["question"], gate=gate, retrieval_weight=weight)
        results.append({"id": g["id"], "question": g["question"], "answerable": g["answerable"], "gold": g["answers"], "prediction": best["answer"],
                        "score": best["score"], "type": best["type"], "doc_id": best["doc_id"], "gold_doc": g["doc_id"],
                        "em": exact_match(best["answer"], g["answers"]) if g["answerable"] else 0.0, "f1": token_f1(best["answer"], g["answers"]) if g["answerable"] else 0.0})
    return results


def forced(results):
    answerable = [r for r in results if r["answerable"]]
    return float(np.mean([r["em"] for r in answerable])), float(np.mean([r["f1"] for r in answerable]))


search = []
for (label, name), pipeline in pipelines.items():
    for gate in (False, True):
        for weight in (0.0, 1.0, 2.0, 4.0):
            em, f1 = forced(evaluate(pipeline, dev, gate, weight))
            search.append({"retriever": label, "reader": name, "gate": gate, "weight": weight, "dev_em": em, "dev_f1": f1})
search = pd.DataFrame(search).sort_values(["dev_em", "dev_f1"], ascending=False)
search.to_csv(RESULTS / "dev_search.csv", index=False)
chosen = search.iloc[0]
print(chosen.to_dict())
best_pipeline = pipelines[(chosen.retriever, chosen.reader)]
dev_results = evaluate(best_pipeline, dev, bool(chosen.gate), float(chosen.weight))
candidates = sorted({r["score"] for r in dev_results if r["score"] > -1e8})
sweep = pd.DataFrame([selective_qa(dev_results, tau) for tau in candidates])
tau = float(sweep.sort_values(["overall_accuracy", "answer_coverage"], ascending=False).iloc[0]["tau"])
print("tau", tau)
search.head(8).round(4)"""),
("md", "## Held-out test: ablation, abstention and answer-type breakdown"),
("code", """ablation = []
def add_row(label, results):
    answerable = [r for r in results if r["answerable"]]
    em, f1 = forced(results)
    lo, hi = bootstrap_ci([r["em"] for r in answerable])
    ablation.append({"system": label, "em": em, "em_lo": lo, "em_hi": hi, "f1": f1})


rule_results = []
for g in test:
    answer, score = rule_answer(g["question"], retriever, MODES[chosen.retriever])
    rule_results.append({"answerable": g["answerable"], "em": exact_match(answer, g["answers"]) if g["answerable"] else 0.0, "f1": token_f1(answer, g["answers"]) if g["answerable"] else 0.0})
add_row("Rule extractor (regex types)", rule_results)
reader_name = chosen.reader
weight = float(chosen.weight)
add_row(f"BM25 + {reader_name}, no gate", evaluate(pipelines[("BM25", reader_name)], test, False, weight))
add_row(f"BM25 + {reader_name}, gate", evaluate(pipelines[("BM25", reader_name)], test, True, weight))
add_row(f"BM25+SGNS + {reader_name}, gate", evaluate(pipelines[("BM25 + SGNS", reader_name)], test, True, weight))
add_row(f"BM25+PA-W2V + {reader_name}, gate", evaluate(pipelines[("BM25 + PA-W2V", reader_name)], test, True, weight))
add_row(f"BM25 rerank PA-W2V 0.25 + {reader_name}, gate", evaluate(pipelines[("BM25 rerank PA-W2V 0.25", reader_name)], test, True, weight))
other = [n for n in READERS if n != reader_name][0]
add_row(f"Dev-selected retriever + {other}, gate", evaluate(pipelines[(chosen.retriever, other)], test, True, weight))
test_results = evaluate(best_pipeline, test, bool(chosen.gate), weight)
add_row("Dev-selected system", test_results)
ablation = pd.DataFrame(ablation)
ablation.to_csv(RESULTS / "test_ablation.csv", index=False)
selective = selective_qa(test_results, tau)
no_abstain = selective_qa(test_results, -1e8)
save_json("test_selective.json", {"chosen": {k: (v.item() if hasattr(v, "item") else v) for k, v in chosen.to_dict().items()}, "tau": tau, "with_abstention": selective, "no_abstention": no_abstain})
pd.DataFrame(test_results).to_json(RESULTS / "test_predictions.jsonl", orient="records", lines=True)
answerable = pd.DataFrame([r for r in test_results if r["answerable"]])
by_type = answerable.groupby("type").agg(questions=("em", "size"), em=("em", "mean"), f1=("f1", "mean")).reset_index()
by_type.to_csv(RESULTS / "test_by_type.csv", index=False)
curve = [selective_qa(test_results, t) for t in sorted({r["score"] for r in test_results if r["score"] > -1e8})]
viz.lines(RESULTS / "qa_abstention.png", [("Answered EM", [c["answer_coverage"] for c in curve], [c["answered_em"] for c in curve]),
                                           ("Unanswerable rejected", [c["answer_coverage"] for c in curve], [c["unanswerable_rejection"] for c in curve])],
          "Abstention trade-off on the held-out questions", "Share of answerable questions answered", "Rate", ylim=(0, 1.05))
ordered = ablation.sort_values("em")
viz.hbar(RESULTS / "qa_ablation.png", ordered["system"].tolist()[::-1], ordered["em"].tolist()[::-1], "Exact match on held-out answerable questions", "Exact match")
display(ablation.round(4), selective, no_abstain, by_type.round(4))"""),
("code", """errors = answerable[answerable.em == 0][["question", "gold", "prediction", "f1", "doc_id", "gold_doc"]]
errors.to_csv(RESULTS / "test_errors.csv", index=False)
errors.head(15)"""),
("md", "## Extra held-out metrics: answer Hit@k, answerability AUROC, rejection F1, error sources"),
("code", """from sklearn.metrics import average_precision_score, roc_auc_score
from telelens.metrics import mcnemar_exact, rejection_scores

answer_hits = {}
gate_flag, gold_answerable = bool(chosen.gate), [g for g in test if g["answerable"]]
for k in (1, 3, 5, 10):
    answer_hits[f"hit@{k}"] = float(np.mean([any(exact_match(a, g["answers"]) for a in best_pipeline.ranked_answers(g["question"], gate_flag, weight)[:k]) for g in gold_answerable]))
    answer_hits[f"f1_hit@{k}"] = float(np.mean([any(token_f1(a, g["answers"]) >= 0.5 for a in best_pipeline.ranked_answers(g["question"], gate_flag, weight)[:k]) for g in gold_answerable]))
labels = [int(r["answerable"]) for r in test_results]
scores = [max(r["score"], -50.0) for r in test_results]
right_doc = answerable["doc_id"] == answerable["gold_doc"]
no_gate_results = evaluate(best_pipeline, test, False, weight)
gate_em = np.array([r["em"] for r in test_results if r["answerable"]])
no_gate_em = np.array([r["em"] for r in no_gate_results if r["answerable"]])
gold_ones = np.ones(len(gate_em))
gate_only, no_gate_only, gate_p = mcnemar_exact(gold_ones, gate_em, no_gate_em)
correct_overall = [(r["em"] if r["score"] >= tau else 0.0) if r["answerable"] else float(r["score"] < tau) for r in test_results]
extra = {"answer_hits": answer_hits, "answerability_auroc": roc_auc_score(labels, scores), "answerability_ap": average_precision_score(labels, scores),
         "rejection": rejection_scores(test_results, tau), "overall_accuracy_ci": bootstrap_ci(correct_overall),
         "right_document_rate": float(right_doc.mean()), "em_given_right_document": float(answerable[right_doc].em.mean()), "em_given_wrong_document": float(answerable[~right_doc].em.mean()),
         "errors_wrong_document": int(((answerable.em == 0) & ~right_doc).sum()), "errors_right_document": int(((answerable.em == 0) & right_doc).sum()),
         "gate_vs_no_gate": {"gate_only_correct": gate_only, "no_gate_only_correct": no_gate_only, "mcnemar_p": gate_p}}
save_json("qa_extra.json", extra)
print(json.dumps(extra, indent=1, default=float))"""),
]

DEMO = [
("md", "# Integrated TeleLens System\n\nOne system, four capabilities: shared embeddings, triage with deferral, digests, and question answering."),
("code", SETUP.format(packages='["rouge-score", "pytest", "sentencepiece", "protobuf"]', topic="integrated")),
("code", """result = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "-p", "no:cacheprovider"], capture_output=True, text=True,
                        env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
print(result.stdout[-1500:], result.stderr[-500:])
(RESULTS / "pytest.txt").write_text(result.stdout, encoding="utf-8")"""),
("code", """from telelens.data import load_comcast, load_telecom
from telelens.pipeline import TeleLens

telecom = load_telecom(10000)
comcast = load_comcast()
system = TeleLens(telecom, comcast, "deepset/roberta-base-squad2", device=DEVICE)
reviews = ["my broadband keeps disconnecting and customer service has not fixed it",
           "the engineer arrived on time and the router was set up in minutes",
           "it was fine i suppose though the price went up after a few months",
           "the speed is good but nobody answers when you call and the bill was wrong once",
           "great deal at first however the router needed replacing twice and they sorted it eventually"]
triage = [{"review": r, **system.triage(r)} for r in reviews]
virgin = telecom[[("virgin" in t.split()) for t in telecom["text"]]].sample(80, random_state=SEED)["text"].tolist()
digest = system.digest(virgin)
questions = ["What was the cancellation fee for the customer who moved to a neighborhood not serviced by Comcast?",
             "Which insurance company refused to pay the full electrician estimate after faulty cable wiring?",
             "How much does Comcast charge for 5G home internet in Seattle?"]
answers = [{"question": q, **system.ask(q)} for q in questions]
neighbours = {w: system.embeddings.neighbours(w, 5) for w in ["reliable", "not_reliable", "unstable"]}
save_json("demo.json", {"triage": triage, "digest": digest, "answers": answers, "neighbours": neighbours})
print(json.dumps({"triage": triage, "digest": digest, "answers": answers, "neighbours": neighbours}, indent=1, default=float)[:6000])"""),
]

NOTEBOOKS = {
    "01_word2vec": W2V,
    "02_sentiment_classification": SENT,
    "03_text_summarization": SUMM,
    "04_factoid_question_answering": QA,
    "05_integrated_system": DEMO,
}


def cell(kind, source, position):
    lines = source.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "id": f"cell-{position}", "metadata": {}, "source": lines}
    return {"cell_type": "code", "id": f"cell-{position}", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def main():
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        notebook = {
            "cells": [cell(kind, source, i) for i, (kind, source) in enumerate(cells)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5,
        }
        (OUT / f"{name}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
