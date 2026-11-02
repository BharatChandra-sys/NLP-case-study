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
