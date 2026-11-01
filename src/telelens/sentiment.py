import numpy as np
import torch
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def grouped_folds(frame, n_splits=5, seed=42):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = frame["text"].factorize()[0]
    return list(splitter.split(frame["text"], frame["label"], groups))


def make_vectorizer(max_features=50000):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True, lowercase=True)


def _classifier(C):
    return CalibratedClassifierCV(LogisticRegression(C=C, max_iter=2000), method="sigmoid", cv=3)


def tfidf_lr(train_texts, y_train, test_texts, C=4.0):
    vectorizer = make_vectorizer()
    x_train, x_test = vectorizer.fit_transform(train_texts), vectorizer.transform(test_texts)
    return _classifier(C).fit(x_train, y_train).predict_proba(x_test)[:, 1]


def dense_lr(x_train, y_train, x_test, C=1.0):
    scaler = StandardScaler().fit(x_train)
    return _classifier(C).fit(scaler.transform(x_train), y_train).predict_proba(scaler.transform(x_test))[:, 1]


def hybrid_lr(train_texts, x_train, y_train, test_texts, x_test, C=4.0, dense_weight=0.15):
    vectorizer = make_vectorizer()
    scaler = StandardScaler().fit(x_train)
    a = sparse.hstack([vectorizer.fit_transform(train_texts), sparse.csr_matrix(scaler.transform(x_train) * dense_weight)]).tocsr()
    b = sparse.hstack([vectorizer.transform(test_texts), sparse.csr_matrix(scaler.transform(x_test) * dense_weight)]).tocsr()
    return _classifier(C).fit(a, y_train).predict_proba(b)[:, 1]


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


def _batch_logits(model, tokenizer, texts, device, max_length, batch_size=128):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    logits = np.zeros((len(texts), 2))
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            ids = order[start : start + batch_size]
            enc = tokenizer([texts[i] for i in ids], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
            logits[ids] = model(**enc).logits.cpu().numpy()
    return logits


def distilbert_finetune(train_texts, y_train, test_texts, device, model_name="distilbert-base-uncased",
                        epochs=2, batch_size=32, lr=2e-5, max_length=256, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit_x, val_x, fit_y, val_y = train_test_split(list(train_texts), list(y_train), test_size=0.1, stratify=y_train, random_state=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * (len(fit_x) // batch_size + 1)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    labels = torch.tensor(fit_y, dtype=torch.long)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(fit_x))
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            enc = tokenizer([fit_x[i] for i in ids], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
            loss = model(**enc, labels=labels[ids].to(device)).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
    temperature = fit_temperature(_batch_logits(model, tokenizer, val_x, device, max_length), val_y, device)
    logits = _batch_logits(model, tokenizer, list(test_texts), device, max_length)
    probs = torch.softmax(torch.tensor(logits) / temperature, dim=1)[:, 1].numpy()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs, temperature


def pretrained_sst2(texts, device, model_name="distilbert-base-uncased-finetuned-sst-2-english", max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    logits = _batch_logits(model, tokenizer, list(texts), device, max_length)
    return torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
