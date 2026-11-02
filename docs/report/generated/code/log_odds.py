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
