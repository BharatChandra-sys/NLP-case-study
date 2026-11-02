def retrieval_scores(ranks, ks=(1, 3, 5, 10, 20)):
    ranks = [r if r is not None else np.inf for r in ranks]
    scores = {f"hit@{k}": float(np.mean([r <= k for r in ranks])) for k in ks}
    scores["mrr"] = float(np.mean([0.0 if np.isinf(r) else 1.0 / r for r in ranks]))
    scores["ndcg@10"] = float(np.mean([1.0 / np.log2(r + 1) if r <= 10 else 0.0 for r in ranks]))
    return scores
