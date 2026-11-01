import numpy as np

from telelens.embeddings import Embeddings, lexical_antonyms, log_odds_polarity, polarity_counterfit, train_sgns
from telelens.metrics import exact_match, expected_calibration_error, js_binary, retrieval_scores, selective_qa, threshold_for_risk, token_f1
from telelens.summarize import pab_mmr, sentence_vectors
from telelens.text import bind_negation, boundary_prf, segment_unpunctuated, split_sentences, tokenize


def test_tokenize_normalises_digits_and_apostrophes():
    assert tokenize("I don't pay $150!") == ["i", "dont", "pay", "<num>"]


def test_bind_negation_skips_intensifiers():
    assert bind_negation("the service is not very reliable".split()) == ["the", "service", "is", "not_reliable"]
    assert bind_negation("not a good deal".split()) == ["not_good", "deal"]
    assert bind_negation("not to".split()) == ["not", "to"]


def test_split_sentences():
    assert split_sentences("It broke. I paid $50! Why?") == ["It broke.", "I paid $50!", "Why?"]


def test_segment_unpunctuated_splits_on_cues_and_keeps_all_words():
    text = "the router arrived on monday morning but the engineer never came and nobody called me back"
    parts = segment_unpunctuated(text)
    assert len(parts) == 2
    assert parts[1].startswith("but")
    assert " ".join(parts) == text


def test_boundary_prf_perfect():
    assert boundary_prf(["a b c", "d e"], ["a b c", "d e"]) == (1.0, 1.0, 1.0)


def test_answer_metrics():
    assert exact_match("the $150", ["$150"]) == 1.0
    assert token_f1("about 4 weeks", ["4 weeks"]) > 0.7
    assert retrieval_scores([1, None])["mrr"] == 0.5


def test_js_binary_bounds():
    assert js_binary(0.5, 0.5) == 0.0
    assert abs(js_binary(1.0, 0.0) - 1.0) < 1e-9


def test_calibration_and_threshold():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.99, 0.99, 0.01, 0.01])
    assert expected_calibration_error(y, prob) < 0.02
    assert threshold_for_risk(y, prob, 0.05) == 0.5


def test_selective_qa_counts_rejections():
    items = [{"answerable": True, "score": 2.0, "em": 1.0, "f1": 1.0}, {"answerable": False, "score": -1.0, "em": 0.0, "f1": 0.0}]
    report = selective_qa(items, 0.0)
    assert report["overall_accuracy"] == 1.0
    assert report["unanswerable_rejection"] == 1.0


def test_log_odds_sign():
    z = log_odds_polarity([["great"] * 20 + ["the"] * 5, ["awful"] * 20 + ["the"] * 5], [1, 0])
    assert z["great"] > 0 > z["awful"]
    assert abs(z["the"]) < 0.1


def test_lexical_antonyms():
    pairs = lexical_antonyms(["stable", "unstable", "good", "not_good", "under"])
    assert ("stable", "unstable") in pairs and ("good", "not_good") in pairs
    assert lexical_antonyms(["stable", "unstable"], excluded={("stable", "unstable")}) == []


def test_counterfit_separates_antonyms_and_keeps_others_near():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=8)
    vectors = np.vstack([shared + 0.01 * rng.normal(size=8), shared + 0.01 * rng.normal(size=8), rng.normal(size=8)])
    emb = Embeddings(["stable", "unstable", "router"], vectors)
    fitted, stats = polarity_counterfit(emb, {"stable": 5.0, "unstable": -5.0}, steps=200, lr=0.05)
    assert stats["repel_pairs"] >= 1
    assert fitted.cosine("stable", "unstable") < emb.cosine("stable", "unstable") - 0.3
    assert fitted.vectors[2] @ emb.vectors[2] > 0.9


def test_train_sgns_learns_cooccurrence():
    docs = [["fast", "speed", "good"] * 5, ["bill", "fee", "charge"] * 5] * 60
    emb, loss = train_sgns(docs, dim=16, window=2, negatives=3, epochs=8, min_count=1, subsample=1.0, batch_size=256, lr=0.01)
    assert loss[-1] < loss[0]
    assert emb.cosine("fast", "speed") > emb.cosine("fast", "bill")


def test_pab_mmr_respects_budget_and_balances_polarity():
    sentences = ["great fast broadband service here"] * 1 + ["great fast broadband service today"] + ["terrible billing and rude staff here", "awful router and slow support line"]
    polarity = np.array([0.95, 0.95, 0.05, 0.05])
    vectors = sentence_vectors(sentences)
    chosen = pab_mmr(vectors, sentences, 11, polarity=polarity, target=0.5, lam=0.5, eta=2.0)
    assert sum(len(sentences[i].split()) for i in chosen) <= 11
    assert {polarity[i] > 0.5 for i in chosen} == {True, False}
