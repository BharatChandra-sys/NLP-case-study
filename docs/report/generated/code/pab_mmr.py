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
