import itertools
from collections import Counter

import networkx as nx
import numpy as np
from rouge_score import rouge_scorer
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .metrics import js_binary

ASPECT_SEEDS = {
    "billing": "bill billing charge charged fee refund payment overcharged credit price".split(),
    "speed": "speed slow fast mbps download buffering disconnect connection outage signal".split(),
    "support": "support service agent advisor rude helpful call hold staff representative".split(),
    "installation": "engineer install installation technician appointment visit router setup activation".split(),
    "contract": "contract cancel cancellation renew deal upgrade switch tariff offer termination".split(),
}
ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def word_count(sentence):
    return len(sentence.split())


def filter_sentences(sentences, min_words=4, max_words=60):
    return [s for s in sentences if min_words <= word_count(s) <= max_words]


def sentence_vectors(sentences):
    return TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1).fit_transform(sentences).toarray()


def _fit_budget(order, sentences, budget):
    chosen, words = [], 0
    for i in order:
        if words + word_count(sentences[i]) <= budget:
            chosen.append(int(i))
            words += word_count(sentences[i])
    return chosen


def lead(sentences, budget):
    return _fit_budget(range(len(sentences)), sentences, budget)


def centroid(vectors, sentences, budget):
    relevance = normalize(vectors) @ normalize(vectors.mean(0, keepdims=True)).ravel()
    return _fit_budget(np.argsort(-relevance), sentences, budget)


def textrank(vectors, sentences, budget):
    sims = np.clip(normalize(vectors) @ normalize(vectors).T, 0, None)
    np.fill_diagonal(sims, 0)
    scores = nx.pagerank(nx.from_numpy_array(sims), weight="weight")
    return _fit_budget(sorted(scores, key=scores.get, reverse=True), sentences, budget)


def pab_mmr(vectors, sentences, budget, polarity=None, aspects=None, target=None, lam=0.7, beta=0.0, eta=0.0):
    x = normalize(vectors)
    relevance = x @ normalize(x.mean(0, keepdims=True)).ravel()
    sims = x @ x.T
    lengths = np.array([word_count(s) for s in sentences], dtype=float)
    aspect_share = {}
    if aspects is not None:
        values, counts = np.unique(aspects, return_counts=True)
        aspect_share = {v: c / counts.sum() for v, c in zip(values, counts)}
    if polarity is not None and target is None:
        target = float(np.average(polarity, weights=lengths))
    chosen, words, seen = [], 0, set()
    candidates = set(range(len(sentences)))
    while candidates:
        best, best_score = None, -np.inf
        for i in sorted(candidates):
            if words + lengths[i] > budget:
                continue
            redundancy_term = sims[i, chosen].max() if chosen else 0.0
            score = lam * relevance[i] - (1 - lam) * redundancy_term
            if aspects is not None and aspects[i] not in seen:
                score += beta * aspect_share.get(aspects[i], 0.0)
            if polarity is not None and eta:
                ids = chosen + [i]
                score -= eta * js_binary(float(np.average(polarity[ids], weights=lengths[ids])), target)
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        chosen.append(best)
        words += lengths[best]
        candidates.remove(best)
        if aspects is not None:
            seen.add(aspects[best])
    return chosen


def kmeans_aspects(vectors, k=4, seed=42):
    k = min(k, len(vectors))
    return KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(vectors)


def expand_aspects(embeddings, top_k=6, min_cosine=0.45):
    lexicon = {}
    for aspect, seeds in ASPECT_SEEDS.items():
        words = set(seeds)
        for seed in seeds:
            for word, score in (embeddings.neighbours(seed, top_k) if seed in embeddings else []):
                if score >= min_cosine and not word.startswith("not_") and len(word) > 2:
                    words.add(word)
        lexicon[aspect] = words
    return lexicon


def lexicon_aspects(sentences, lexicon):
    labels = []
    for sentence in sentences:
        tokens = sentence.lower().split()
        counts = {a: sum(t in words for t in tokens) for a, words in lexicon.items()}
        best = max(counts, key=counts.get)
        labels.append(best if counts[best] else "other")
    return np.array(labels)


def rouge_multi(summary, references):
    scores = [ROUGE.score(ref, summary) for ref in references]
    output = {k: float(np.mean([s[k].fmeasure for s in scores])) for k in ("rouge1", "rouge2", "rougeL")}
    for k in ("rouge1", "rouge2"):
        output[f"{k}_p"] = float(np.mean([s[k].precision for s in scores]))
        output[f"{k}_r"] = float(np.mean([s[k].recall for s in scores]))
    return output


def unigram_js(summary, source):
    a, b = Counter(summary.lower().split()), Counter(source.lower().split())
    vocab = sorted(set(a) | set(b))
    return float(jensenshannon([a[w] for w in vocab], [b[w] for w in vocab], base=2) ** 2)


def weighted_polarity(polarity, sentences, chosen):
    lengths = np.array([word_count(sentences[i]) for i in chosen], dtype=float)
    return float(np.average(polarity[chosen], weights=lengths)) if len(chosen) else 0.5


def redundancy(vectors, chosen):
    if len(chosen) < 2:
        return 0.0
    x = normalize(vectors[chosen])
    return float(np.mean([x[a] @ x[b] for a, b in itertools.combinations(range(len(chosen)), 2)]))


def aspect_recall(aspects, chosen):
    values, counts = np.unique(aspects, return_counts=True)
    covered = set(aspects[chosen])
    return float(sum(c for v, c in zip(values, counts) if v in covered) / counts.sum())


def polarity_gap(polarity, sentences, chosen, target):
    if not len(chosen):
        return 1.0
    lengths = np.array([word_count(sentences[i]) for i in chosen], dtype=float)
    return js_binary(float(np.average(polarity[chosen], weights=lengths)), target)


def bart_summaries(documents, device, model_name="facebook/bart-large-cnn", batch_size=8, max_length=90, min_length=30):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    output = []
    for start in range(0, len(documents), batch_size):
        enc = tokenizer(documents[start : start + batch_size], truncation=True, max_length=1024, padding=True, return_tensors="pt").to(device)
        ids = model.generate(**enc, num_beams=4, max_length=max_length, min_length=min_length)
        output.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    del model
    return output


def novel_token_rate(summary, source):
    source_tokens = set(source.lower().split())
    tokens = summary.lower().split()
    return sum(t not in source_tokens for t in tokens) / max(len(tokens), 1)
