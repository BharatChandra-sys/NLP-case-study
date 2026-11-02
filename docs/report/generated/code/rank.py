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
