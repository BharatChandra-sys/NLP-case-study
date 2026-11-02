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
