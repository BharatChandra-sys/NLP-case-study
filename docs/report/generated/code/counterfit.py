def polarity_counterfit(emb, polarity, excluded_pairs=frozenset(), z_threshold=2.58, k=10,
                        repel_margin=0.0, attract_margin=0.6, repel_weight=1.0, attract_weight=0.3,
                        anchor_weight=1.0, steps=300, lr=0.01, device="cpu"):
    antonyms = lexical_antonyms(emb.words, excluded_pairs)
    attract, repel = polar_neighbour_constraints(emb, polarity, z_threshold, k)
    held_out = {tuple(sorted(p)) for p in excluded_pairs}
    attract = [p for p in attract if p not in held_out]
    repel = sorted({p for p in repel if p not in held_out} | set(antonyms))
    original = torch.tensor(emb.vectors, dtype=torch.float, device=device)
    vectors = original.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([vectors], lr=lr)

    def ids(pairs):
        if not pairs:
            return None, None
        arr = torch.tensor([(emb.index[a], emb.index[b]) for a, b in pairs], device=device)
        return arr[:, 0], arr[:, 1]

    r_a, r_b = ids(repel)
    s_a, s_b = ids(attract)
    for _ in range(steps):
        normed = F.normalize(vectors, dim=1)
        loss = anchor_weight * (normed - original).pow(2).sum(1).mean()
        if r_a is not None:
            loss = loss + repel_weight * F.relu((normed[r_a] * normed[r_b]).sum(1) - repel_margin).mean()
        if s_a is not None:
            loss = loss + attract_weight * F.relu(attract_margin - (normed[s_a] * normed[s_b]).sum(1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    result = Embeddings(emb.words, F.normalize(vectors, dim=1).detach().cpu().numpy(), bound=emb.bound)
    stats = {"antonym_pairs": len(antonyms), "repel_pairs": len(repel), "attract_pairs": len(attract)}
    return result, stats
