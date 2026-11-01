import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = "oliverguhr/fullstop-punctuation-multilang-large"
BOUNDARY = {".", "?", "!"}


class PunctuationRestorer:
    def __init__(self, device, model_name=MODEL, window=200, batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device).eval()
        self.labels = self.model.config.id2label
        self.device = device
        self.window = window
        self.batch_size = batch_size

    def _boundaries(self, chunks):
        flags = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            enc = self.tokenizer(batch, is_split_into_words=True, truncation=True, padding=True, return_tensors="pt")
            word_ids = [enc.word_ids(i) for i in range(len(batch))]
            with torch.no_grad():
                predicted = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.argmax(-1).cpu().numpy()
            for i, words in enumerate(batch):
                last = {}
                for position, word in enumerate(word_ids[i]):
                    if word is not None:
                        last[word] = self.labels[int(predicted[i, position])]
                flags.append([last.get(w, "0") in BOUNDARY for w in range(len(words))])
        return flags

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
