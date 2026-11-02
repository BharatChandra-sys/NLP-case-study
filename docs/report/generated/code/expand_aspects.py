def expand_aspects(embeddings, top_k=6, min_cosine=0.45):
    lexicon = {}
    for aspect, seeds in ASPECT_SEEDS.items():
        words = set(seeds)
        for seed in seeds:
            for word, score in (embeddings.neighbours(seed, top_k) if seed in embeddings else []):
                if score >= min_cosine and not word.startswith("not_") and len(word) > 2:
                    words.add(word)
        lexicon[aspect] = words
    return lexicon
