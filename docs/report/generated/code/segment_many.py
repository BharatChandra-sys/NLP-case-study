def segment_many(self, texts):
    chunks, owners = [], []
    for owner, text in enumerate(texts):
        words = text.split()
        for start in range(0, len(words), self.window):
            chunks.append(words[start : start + self.window])
            owners.append(owner)
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
    ordered = self._boundaries([chunks[i] for i in order])
    flags = {chunk: ordered[position] for position, chunk in enumerate(order)}
    output = [[] for _ in texts]
    current = [[] for _ in texts]
    for chunk_index in range(len(chunks)):
        owner = owners[chunk_index]
        for word, boundary in zip(chunks[chunk_index], flags[chunk_index]):
            current[owner].append(word)
            if boundary:
                output[owner].append(" ".join(current[owner]))
                current[owner] = []
    for owner, rest in enumerate(current):
        if rest:
            output[owner].append(" ".join(rest))
    return output
