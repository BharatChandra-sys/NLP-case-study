import math
import re
from collections import Counter

import numpy as np
import torch
from scipy import sparse
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from .text import STOPWORDS, bind_negation, split_sentences, tokenize

_RAW = re.compile(r"[a-z0-9$&]+(?:[./:-][a-z0-9]+)*")
_NUMBERS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|ninety|hundred"
_UNITS = "second|sec|minute|min|hour|hr|day|week|wk|month|mo|year|yr"
GATES = {
    "MONEY": re.compile(r"\$|\d|dollar", re.I),
    "DURATION": re.compile(rf"(\d|\b(?:{_NUMBERS})\b).*\b(?:{_UNITS})s?\b", re.I),
    "COUNT": re.compile(rf"\d|\b(?:{_NUMBERS})\b", re.I),
    "TIME": re.compile(r"\d\s*:?\d*\s*(?:am|pm)|\d:\d\d", re.I),
    "DATE": re.compile(r"\d{1,2}[/-]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d|\b(?:19|20)\d\d\b", re.I),
}
_CANDIDATES = {
    "MONEY": re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:/\w+)?|\b\d[\d,]*(?:\.\d+)?\s?dollars?\b|\b\d+\.\d\d\b", re.I),
    "DURATION": re.compile(rf"\b(?:\d[\d./ ]*|(?:{_NUMBERS})\s)\s*(?:to\s+\d+\s+)?(?:{_UNITS})s?\b", re.I),
    "COUNT": re.compile(rf"\b\d+\b|\b(?:{_NUMBERS})\b", re.I),
    "TIME": re.compile(r"\b\d{1,2}(?::\d\d)?\s?(?:am|pm)\b", re.I),
    "DATE": re.compile(r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan\.?|feb\.?|sept?\.?|oct\.?|nov\.?|dec\.?)\s+\d{1,2}(?:,\s*\d{4})?|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", re.I),
    "ENTITY": re.compile(r"\b[A-Z][\w&-]*(?:[ -][A-Z][\w&-]*)*\b"),
}


def retrieval_tokens(text):
    return [t for t in _RAW.findall(str(text).lower()) if t not in STOPWORDS]


def dense_tokens(text):
    return bind_negation(tokenize(text))


def answer_type(question):
    q = question.lower()
    if re.search(r"\bat what time\b|\bwhat time\b", q):
        return "TIME"
    if re.search(r"\bwhat date\b|\bwhich date\b|\bwhen did\b|\bearliest date\b", q):
        return "DATE"
    if re.search(r"how long|how far|how often|how many (?:seconds|minutes|hours|weeks|months|years|days)|minimum hold|wait time", q):
        return "DURATION"
    if re.search(r"how much|what (?:\w+ ){0,3}(?:price|fee|charge|bill|deposit|credit|refund|rate|amount|cost|rebate)\b|what did .* (?:cost|rise|increase)", q):
        return "MONEY"
    if re.search(r"how many|how old", q):
        return "COUNT"
    return "ENTITY"


def passes_gate(kind, span):
    gate = GATES.get(kind)
    return bool(gate.search(span)) if gate else True


def make_passages(frame, window=3, stride=2):
    passages = []
    for doc_id, text in zip(frame["doc_id"], frame["text"]):
        sentences = split_sentences(text) or [text]
        for start in range(0, max(len(sentences) - window + 1, 1), stride):
            passages.append({"doc_id": int(doc_id), "text": " ".join(sentences[start : start + window])})
    return passages


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        counts = [Counter(d) for d in documents]
        self.vocab = {t: i for i, t in enumerate(sorted({t for c in counts for t in c}))}
        lengths = np.array([sum(c.values()) for c in counts], dtype=float)
        avg = max(lengths.mean(), 1.0)
        rows, cols, values = [], [], []
        df = Counter()
        for row, c in enumerate(counts):
            df.update(c.keys())
            norm = k1 * (1 - b + b * lengths[row] / avg)
            for term, tf in c.items():
                rows.append(row)
                cols.append(self.vocab[term])
                values.append(tf * (k1 + 1) / (tf + norm))
        self.matrix = sparse.csc_matrix((values, (rows, cols)), shape=(len(counts), len(self.vocab)))
        n = len(counts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, tokens):
        total = np.zeros(self.matrix.shape[0])
        for term in tokens:
            if term in self.vocab:
                total += self.idf[term] * self.matrix[:, self.vocab[term]].toarray().ravel()
        return total


class HybridRetriever:
    def __init__(self, passages, spaces=None, rrf_k=60):
        self.passages = passages
        self.rrf_k = rrf_k
        self.bm25 = BM25([retrieval_tokens(p["text"]) for p in passages])
        self.spaces = {}
        for name, emb in (spaces or {}).items():
            tokens = [dense_tokens(p["text"]) if emb.bound else tokenize(p["text"]) for p in passages]
            df = Counter(t for d in tokens for t in set(d))
            weights = {t: math.log(len(tokens) / (1 + f)) + 1 for t, f in df.items()}
            self.spaces[name] = (emb, weights, emb.embed_many(tokens, weights))

    def _dense(self, question, name):
        emb, weights, matrix = self.spaces[name]
        q = emb.embed(dense_tokens(question) if emb.bound else tokenize(question), weights)
        return matrix @ q

    def rank(self, question, mode="bm25", k=100):
        if mode == "bm25":
            return [int(i) for i in np.argsort(-self.bm25.scores(retrieval_tokens(question)))[:k]]
        if mode in self.spaces:
            return [int(i) for i in np.argsort(-self._dense(question, mode))[:k]]
        if mode.startswith("rerank:"):
            _, dense, gamma = mode.split(":")
            lexical = self.bm25.scores(retrieval_tokens(question))
            top = np.argsort(-lexical)[:100]
            semantic = self._dense(question, dense)[top]
            z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            order = np.argsort(-(z(lexical[top]) + float(gamma) * z(semantic)))
            return [int(i) for i in top[order][:k]]
        dense = mode.split("+")[1]
        a = np.argsort(-self.bm25.scores(retrieval_tokens(question)))[:500]
        b = np.argsort(-self._dense(question, dense))[:500]
        fused = Counter()
        for ranking in (a, b):
            for position, passage in enumerate(ranking):
                fused[int(passage)] += 1.0 / (self.rrf_k + position + 1)
        return [p for p, _ in fused.most_common(k)]

    def doc_rank(self, ranked, doc_id):
        seen = []
        for passage in ranked:
            doc = self.passages[passage]["doc_id"]
            if doc not in seen:
                seen.append(doc)
            if doc == doc_id:
                return len(seen)
        return None


class Reader:
    def __init__(self, model_name, device, max_length=384):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_length = max_length

    def spans(self, questions, contexts, top=20, max_answer_tokens=12, keep=8, batch_size=64):
        results = []
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            batch_q = questions[start : start + batch_size]
            enc = self.tokenizer(batch_q, batch, truncation="only_second", max_length=self.max_length,
                                 padding=True, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").numpy()
            type_mask = np.array([[s == 1 for s in enc.sequence_ids(i)] for i in range(len(batch))])
            inputs = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            with torch.no_grad():
                out = self.model(**inputs)
            s_all, e_all = out.start_logits.cpu().numpy(), out.end_logits.cpu().numpy()
            for i, context in enumerate(batch):
                cls = s_all[i, 0] + e_all[i, 0]
                valid = np.where(type_mask[i])[0]
                found = []
                if len(valid):
                    s_ids = valid[np.argsort(-s_all[i, valid])[:top]]
                    e_ids = valid[np.argsort(-e_all[i, valid])[:top]]
                    for s in s_ids:
                        for e in e_ids:
                            if s <= e < s + max_answer_tokens:
                                a, b = offsets[i, s, 0], offsets[i, e, 1]
                                if b > a:
                                    raw = float(s_all[i, s] + e_all[i, e])
                                    found.append({"text": context[a:b].strip(), "raw": raw, "margin": raw - float(cls)})
                found.sort(key=lambda x: -x["raw"])
                results.append(found[:keep])
        return results


class QAPipeline:
    def __init__(self, retriever, reader, mode="bm25", top_k=10):
        self.retriever = retriever
        self.reader = reader
        self.mode = mode
        self.top_k = top_k
        self.cache = {}

    def read(self, question):
        if question not in self.cache:
            ranked = self.retriever.rank(question, self.mode, 100)
            top = ranked[: self.top_k]
            contexts = [self.retriever.passages[p]["text"] for p in top]
            self.cache[question] = (ranked, top, contexts, self.reader.spans([question] * len(contexts), contexts))
        return self.cache[question]

    def answer(self, question, gate=True, retrieval_weight=2.0):
        ranked, top, contexts, all_spans = self.read(question)
        kind = answer_type(question)
        best = {"answer": "", "score": -1e9, "doc_id": None, "evidence": "", "type": kind}
        for position, spans in enumerate(all_spans):
            for span in spans:
                if gate and not passes_gate(kind, span["text"]):
                    continue
                score = span["margin"] + retrieval_weight / (position + 1)
                if score > best["score"]:
                    best = {"answer": span["text"], "score": score, "doc_id": self.retriever.passages[top[position]]["doc_id"],
                            "evidence": contexts[position], "type": kind}
        return best, ranked


def rule_answer(question, retriever, mode="bm25", top_k=5):
    kind = answer_type(question)
    ranked = retriever.rank(question, mode, 100)
    terms = set(retrieval_tokens(question))
    question_lower = question.lower()
    best, best_score = "", -1e9
    for position, passage in enumerate(ranked[:top_k]):
        text = retriever.passages[passage]["text"]
        hits = [m.start() for t in terms for m in re.finditer(re.escape(t), text.lower())]
        for match in _CANDIDATES.get(kind, _CANDIDATES["ENTITY"]).finditer(text):
            span = match.group().strip()
            if span.lower() in question_lower:
                continue
            distance = min((abs(match.start() - h) for h in hits), default=500)
            score = -distance / 50.0 - position
            if score > best_score:
                best, best_score = span, score
    return best, best_score
