def threshold_for_risk(y_true, prob, target_risk):
    rows = risk_coverage(y_true, prob)
    valid = [r for r in rows if r["risk"] <= target_risk]
    return max(valid, key=lambda r: r["coverage"])["tau"] if valid else 1.01
