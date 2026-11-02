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
