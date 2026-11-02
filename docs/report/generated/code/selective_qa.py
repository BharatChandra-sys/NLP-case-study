def selective_qa(items, tau):
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]
    answered = [i for i in answerable if i["score"] >= tau]
    correct = sum(i["em"] for i in answered)
    false_answers = sum(i["score"] >= tau for i in unanswerable)
    return {
        "tau": float(tau),
        "overall_accuracy": (correct + len(unanswerable) - false_answers) / max(len(items), 1),
        "answer_coverage": len(answered) / max(len(answerable), 1),
        "answered_em": correct / max(len(answered), 1),
        "answered_f1": float(np.mean([i["f1"] for i in answered])) if answered else 0.0,
        "unanswerable_rejection": 1 - false_answers / max(len(unanswerable), 1),
    }
