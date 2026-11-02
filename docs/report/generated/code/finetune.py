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
