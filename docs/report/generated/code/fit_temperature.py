def fit_temperature(logits, labels, device):
    logits = torch.tensor(logits, dtype=torch.float, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())
