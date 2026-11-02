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
