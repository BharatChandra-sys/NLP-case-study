import re

NEGATORS = frozenset(
    "not no never cannot dont didnt doesnt wont cant isnt wasnt arent werent "
    "couldnt wouldnt shouldnt havent hasnt hadnt".split()
)
SKIP_AFTER_NEGATOR = frozenset("a an the very really so too quite that at all even be been being".split())
STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been before being below "
    "between both but by can could did do does doing down during each few for from further had has have "
    "having he her here hers him his how i if im in into is it its ive me more most my myself of off on "
    "once only or other our ours out over own same she should so some such than that the their theirs them "
    "then there these they this those through to too under until up us very was we were what when where "
    "which while who whom why will with would you your yours".split()
)
CUE_WORDS = frozenset(
    "but however although though so then after when because since while still also yet "
    "anyway overall unfortunately finally eventually now instead otherwise".split()
)
_TOKEN = re.compile(r"[a-z0-9$£%']+")
_SENTENCE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")


def tokenize(text):
    tokens = []
    for raw in _TOKEN.findall(str(text).lower().replace("’", "'")):
        token = raw.replace("'", "")
        if not token:
            continue
        tokens.append("<num>" if any(ch.isdigit() for ch in token) else token)
    return tokens


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


def is_punctuated(text):
    return bool(re.search(r"[.!?]", str(text)))


def split_sentences(text):
    return [s.strip() for s in _SENTENCE.split(str(text).strip()) if s.strip()]


def segment_unpunctuated(text, min_words=6, max_words=24):
    words = str(text).split()
    segments, current = [], []
    for word in words:
        cue = word.lower() in CUE_WORDS
        if current and ((cue and len(current) >= min_words) or len(current) >= max_words):
            segments.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        if segments and len(current) < min_words:
            segments[-1] += " " + " ".join(current)
        else:
            segments.append(" ".join(current))
    return segments


def segment_fixed(text, size=15):
    words = str(text).split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def segment(text):
    return split_sentences(text) if is_punctuated(text) else segment_unpunctuated(text)


def boundary_positions(segments):
    positions, total = set(), 0
    for seg in segments[:-1]:
        total += len(seg.split())
        positions.add(total)
    return positions


def boundary_prf(true_segments, predicted_segments, tolerance=2):
    gold, pred = boundary_positions(true_segments), boundary_positions(predicted_segments)
    tp_pred = sum(any(abs(p - g) <= tolerance for g in gold) for p in pred)
    tp_gold = sum(any(abs(p - g) <= tolerance for p in pred) for g in gold)
    precision = tp_pred / len(pred) if pred else 0.0
    recall = tp_gold / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
