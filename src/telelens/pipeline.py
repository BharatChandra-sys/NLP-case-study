import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from .embeddings import log_odds_polarity, polarity_counterfit, train_sgns
from .punctuate import PunctuationRestorer
from .qa import HybridRetriever, QAPipeline, Reader, make_passages
from .sentiment import make_vectorizer
from .summarize import expand_aspects, filter_sentences, lexicon_aspects, pab_mmr, sentence_vectors
from .text import bind_negation, is_punctuated, split_sentences, tokenize


def build_polar_embeddings(texts, labels, device="cpu", excluded_pairs=frozenset(), **params):
    tokens = [bind_negation(tokenize(t)) for t in texts]
    base, _ = train_sgns(tokens, device=device, bound=True)
    known = [(d, y) for d, y in zip(tokens, labels) if y in (0, 1)]
    polarity = log_odds_polarity([d for d, _ in known], [y for _, y in known])
    polar, stats = polarity_counterfit(base, polarity, excluded_pairs=excluded_pairs, device=device, **params)
    return base, polar, polarity, stats


class TeleLens:
    def __init__(self, train_frame, corpus_frame, reader_name, device="cpu", defer_threshold=0.735,
                 qa_threshold=3.272, retrieval_mode="rerank:sgns:0.25", retrieval_weight=2.0):
        labels = list(train_frame["label"]) + [-1] * len(corpus_frame)
        texts = list(train_frame["text"]) + list(corpus_frame["text"])
        _, self.embeddings, _, _ = build_polar_embeddings(texts, labels, device=device)
        self.sgns, _ = train_sgns([tokenize(t) for t in texts], device=device)
        self.vectorizer = make_vectorizer()
        x = self.vectorizer.fit_transform(train_frame["text"])
        self.classifier = CalibratedClassifierCV(LogisticRegression(C=4.0, max_iter=2000), method="sigmoid", cv=3)
        self.classifier.fit(x, train_frame["label"])
        self.history_texts = list(train_frame["text"])
        self.history_labels = np.array(train_frame["label"])
        self.history_vectors = normalize(x)
        self.defer_threshold = defer_threshold
        self.qa_threshold = qa_threshold
        self.retrieval_weight = retrieval_weight
        self.lexicon = expand_aspects(self.embeddings)
        retriever = HybridRetriever(make_passages(corpus_frame), {"pa_w2v": self.embeddings, "sgns": self.sgns})
        self.qa = QAPipeline(retriever, Reader(reader_name, device), mode=retrieval_mode)
        self.restorer = PunctuationRestorer(device)

    def good_probability(self, texts):
        return self.classifier.predict_proba(self.vectorizer.transform(texts))[:, 1]

    def triage(self, review, k=3):
        prob = float(self.good_probability([review])[0])
        confidence = max(prob, 1 - prob)
        vector = normalize(self.vectorizer.transform([review]))
        scores = (self.history_vectors @ vector.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return {
            "prediction": "Good" if prob >= 0.5 else "Bad",
            "probability_good": prob,
            "decision": "automatic" if confidence >= self.defer_threshold else "defer_to_analyst",
            "similar": [{"cosine": float(scores[i]), "label": "Good" if self.history_labels[i] else "Bad",
                         "text": self.history_texts[i][:160]} for i in order],
        }

    def digest(self, reviews, budget=60, lam=0.7, beta=0.3, eta=0.5):
        unpunctuated = [r for r in reviews if not is_punctuated(r)]
        pieces = [s for r in reviews if is_punctuated(r) for s in split_sentences(r)]
        pieces += [s for part in self.restorer.segment_many(unpunctuated) for s in part]
        sentences = filter_sentences(pieces, min_words=5)
        polarity = self.good_probability(sentences)
        target = float(np.mean(self.good_probability(reviews) >= 0.5))
        chosen = pab_mmr(sentence_vectors(sentences), sentences, budget, polarity=polarity,
                         aspects=lexicon_aspects(sentences, self.lexicon), target=target, lam=lam, beta=beta, eta=eta)
        return {"share_good": target, "summary": [sentences[i] for i in chosen]}

    def ask(self, question):
        best, _ = self.qa.answer(question, gate=True, retrieval_weight=self.retrieval_weight)
        if best["score"] < self.qa_threshold:
            return {"answer": None, "status": "abstained", "score": best["score"]}
        return {"answer": best["answer"], "status": "answered", "score": best["score"],
                "doc_id": best["doc_id"], "evidence": best["evidence"], "type": best["type"]}
