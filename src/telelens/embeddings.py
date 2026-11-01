from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

ANTONYM_PREFIXES = ("un", "in", "dis", "im", "ir")


class Embeddings:
    def __init__(self, words, vectors, bound=False):
        self.bound = bound
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-9)

    def __contains__(self, word):
        return word in self.index

    def cosine(self, a, b):
        if a not in self.index or b not in self.index:
            return None
        return float(self.vectors[self.index[a]] @ self.vectors[self.index[b]])

    def neighbours(self, word, k=10):
        if word not in self.index:
            return []
        scores = self.vectors @ self.vectors[self.index[word]]
        order = np.argsort(-scores)
        return [(self.words[i], float(scores[i])) for i in order if self.words[i] != word][:k]

    def embed(self, tokens, weights=None):
        ids = [self.index[t] for t in tokens if t in self.index]
        if not ids:
            return np.zeros(self.vectors.shape[1], dtype=np.float32)
        w = np.ones(len(ids), dtype=np.float32) if weights is None else np.array(
            [weights.get(self.words[i], 1.0) for i in ids], dtype=np.float32
        )
        vec = (self.vectors[ids] * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_many(self, documents, weights=None):
        return np.vstack([self.embed(d, weights) for d in documents]) if documents else np.zeros((0, self.vectors.shape[1]))

    def save(self, path):
        np.savez_compressed(path, words=np.array(self.words), vectors=self.vectors)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls(data["words"].tolist(), data["vectors"])


def _skipgram_pairs(ids_docs, window, keep_prob, rng):
    centres, contexts = [], []
    for doc in ids_docs:
        ids = doc[rng.random(len(doc)) < keep_prob[doc]]
        n = len(ids)
        if n < 2:
            continue
        reach = rng.integers(1, window + 1, size=n)
        for i in range(n):
            lo, hi = max(0, i - reach[i]), min(n, i + reach[i] + 1)
            for j in range(lo, hi):
                if j != i:
                    centres.append(ids[i])
                    contexts.append(ids[j])
    return np.array(centres, dtype=np.int64), np.array(contexts, dtype=np.int64)


def train_sgns(documents, dim=200, window=5, negatives=10, epochs=5, min_count=5, subsample=1e-3,
               lr=0.002, batch_size=8192, seed=42, device="cpu", bound=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    counts = Counter(t for d in documents for t in d)
    words = [w for w, c in counts.most_common() if c >= min_count]
    index = {w: i for i, w in enumerate(words)}
    freq = np.array([counts[w] for w in words], dtype=np.float64)
    probs = freq / freq.sum()
    keep = np.minimum(1.0, np.sqrt(subsample / probs) + subsample / probs)
    noise = torch.tensor(freq**0.75 / (freq**0.75).sum(), dtype=torch.float, device=device)
    ids_docs = [np.array([index[t] for t in d if t in index], dtype=np.int64) for d in documents]

    centre = torch.nn.Embedding(len(words), dim, sparse=True, device=device)
    context = torch.nn.Embedding(len(words), dim, sparse=True, device=device)
    torch.nn.init.uniform_(centre.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(context.weight)
    optimizer = torch.optim.SparseAdam([*centre.parameters(), *context.parameters()], lr=lr)

    history = []
    for _ in range(epochs):
        c_np, o_np = _skipgram_pairs(ids_docs, window, keep, rng)
        order = rng.permutation(len(c_np))
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            sel = order[start : start + batch_size]
            c = torch.as_tensor(c_np[sel], device=device)
            o = torch.as_tensor(o_np[sel], device=device)
            n = torch.multinomial(noise, len(sel) * negatives, replacement=True).view(len(sel), negatives)
            vc = centre(c)
            pos = (vc * context(o)).sum(1)
            neg = torch.bmm(context(n), vc.unsqueeze(2)).squeeze(2)
            loss = -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        history.append(total / max(batches, 1))
    return Embeddings(words, centre.weight.detach().cpu().numpy(), bound=bound), history


def log_odds_polarity(documents, labels, prior=0.01):
    pos, neg = Counter(), Counter()
    for doc, y in zip(documents, labels):
        (pos if y == 1 else neg).update(doc)
    vocab = set(pos) | set(neg)
    a0 = prior * len(vocab)
    n_pos, n_neg = sum(pos.values()), sum(neg.values())
    z = {}
    for w in vocab:
        yp, yn = pos[w], neg[w]
        delta = np.log((yp + prior) / (n_pos + a0 - yp - prior)) - np.log((yn + prior) / (n_neg + a0 - yn - prior))
        z[w] = float(delta / np.sqrt(1 / (yp + prior) + 1 / (yn + prior)))
    return z


def lexical_antonyms(words, excluded=frozenset()):
    vocab = set(words)
    pairs = set()
    for w in vocab:
        if w.startswith("not_") and w[4:] in vocab:
            pairs.add((w[4:], w))
        for prefix in ANTONYM_PREFIXES:
            if w.startswith(prefix) and len(w) - len(prefix) >= 4 and w[len(prefix):] in vocab:
                pairs.add((w[len(prefix):], w))
    return sorted(p for p in pairs if p not in excluded and (p[1], p[0]) not in excluded)


def polar_neighbour_constraints(emb, polarity, z_threshold=2.58, k=10):
    polar = [w for w in emb.words if abs(polarity.get(w, 0.0)) >= z_threshold]
    ids = [emb.index[w] for w in polar]
    if not ids:
        return [], []
    sims = emb.vectors[ids] @ emb.vectors[ids].T
    np.fill_diagonal(sims, -1.0)
    attract, repel = set(), set()
    for row, w in enumerate(polar):
        for col in np.argsort(-sims[row])[:k]:
            other = polar[col]
            pair = tuple(sorted((w, other)))
            (attract if polarity[w] * polarity[other] > 0 else repel).add(pair)
    return sorted(attract), sorted(repel)


def polarity_counterfit(emb, polarity, excluded_pairs=frozenset(), z_threshold=2.58, k=10,
                        repel_margin=0.0, attract_margin=0.6, repel_weight=1.0, attract_weight=0.3,
                        anchor_weight=1.0, steps=300, lr=0.01, device="cpu"):
    antonyms = lexical_antonyms(emb.words, excluded_pairs)
    attract, repel = polar_neighbour_constraints(emb, polarity, z_threshold, k)
    held_out = {tuple(sorted(p)) for p in excluded_pairs}
    attract = [p for p in attract if p not in held_out]
    repel = sorted({p for p in repel if p not in held_out} | set(antonyms))
    original = torch.tensor(emb.vectors, dtype=torch.float, device=device)
    vectors = original.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([vectors], lr=lr)

    def ids(pairs):
        if not pairs:
            return None, None
        arr = torch.tensor([(emb.index[a], emb.index[b]) for a, b in pairs], device=device)
        return arr[:, 0], arr[:, 1]

    r_a, r_b = ids(repel)
    s_a, s_b = ids(attract)
    for _ in range(steps):
        normed = F.normalize(vectors, dim=1)
        loss = anchor_weight * (normed - original).pow(2).sum(1).mean()
        if r_a is not None:
            loss = loss + repel_weight * F.relu((normed[r_a] * normed[r_b]).sum(1) - repel_margin).mean()
        if s_a is not None:
            loss = loss + attract_weight * F.relu(attract_margin - (normed[s_a] * normed[s_b]).sum(1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    result = Embeddings(emb.words, F.normalize(vectors, dim=1).detach().cpu().numpy(), bound=emb.bound)
    stats = {"antonym_pairs": len(antonyms), "repel_pairs": len(repel), "attract_pairs": len(attract)}
    return result, stats
