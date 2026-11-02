def risk_coverage(y_true, prob, grid=None):
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    confidence = np.maximum(prob, 1 - prob)
    correct = (prob >= 0.5).astype(int) == y_true
    grid = np.linspace(0.5, 0.995, 100) if grid is None else grid
    rows = []
    for tau in grid:
        kept = confidence >= tau
        if kept.sum():
            rows.append({"tau": float(tau), "coverage": float(kept.mean()), "risk": float(1 - correct[kept].mean())})
    return rows
