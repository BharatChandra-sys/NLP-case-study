def train_sgns(documents, dim=200, window=5, negatives=10, epochs=5, min_count=5, subsample=1e-3,
               lr=0.002, batch_size=8192, seed=42, device="cpu", bound=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    counts = Counter(t for d in documents for t in d)
    words = [w for w, c in counts.most_common() if c >= min_count]
    index = {w: i for i, w in enumerate(words)}
    freq = np.array([counts[w] for w in words], dtype=np.float64)
    probs = freq / freq.sum()
    keep = np.minimum(1.0, np.sqrt(subsample / probs) + subsample / probs)
    noise = torch.tensor(freq**0.75 / (freq**0.75).sum(), dtype=torch.float, device=device)
    ids_docs = [np.array([index[t] for t in d if t in index], dtype=np.int64) for d in documents]

    centre = torch.nn.Embedding(len(words), dim, sparse=True, device=device)
    context = torch.nn.Embedding(len(words), dim, sparse=True, device=device)
    torch.nn.init.uniform_(centre.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(context.weight)
    optimizer = torch.optim.SparseAdam([*centre.parameters(), *context.parameters()], lr=lr)

    history = []
    for _ in range(epochs):
        c_np, o_np = _skipgram_pairs(ids_docs, window, keep, rng)
        order = rng.permutation(len(c_np))
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            sel = order[start : start + batch_size]
            c = torch.as_tensor(c_np[sel], device=device)
            o = torch.as_tensor(o_np[sel], device=device)
            n = torch.multinomial(noise, len(sel) * negatives, replacement=True).view(len(sel), negatives)
            vc = centre(c)
            pos = (vc * context(o)).sum(1)
            neg = torch.bmm(context(n), vc.unsqueeze(2)).squeeze(2)
            loss = -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        history.append(total / max(batches, 1))
    return Embeddings(words, centre.weight.detach().cpu().numpy(), bound=bound), history
