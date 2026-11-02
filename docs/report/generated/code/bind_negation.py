def bind_negation(tokens):
    bound = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in NEGATORS:
            j = i + 1
            while j < len(tokens) and tokens[j] in SKIP_AFTER_NEGATOR:
                j += 1
            if j < len(tokens) and tokens[j] not in STOPWORDS and tokens[j] not in NEGATORS:
                bound.append(f"not_{tokens[j]}")
                i = j + 1
                continue
        bound.append(token)
        i += 1
    return bound
