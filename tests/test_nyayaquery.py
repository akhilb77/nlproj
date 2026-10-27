"""Unit + end-to-end tests.  Run:  python -m pytest -v"""

import json

import pytest

from src.cyk_parser import CYKParser
from src.document_loader import load_documents
from src.hmm_pos_tagger import HMMPosTagger, signature
from src.information_retrieval import TfidfRetriever
from src.question_answering import NyayaQuery
from src.regex_analyzer import extract_clause, find_references, mask_references
from src.text_preprocessor import analyze, tokenize


@pytest.fixture(scope="session")
def qa():
    return NyayaQuery()


# ---------------- loader / preprocessing ----------------
def test_loader_reads_constitution():
    recs = load_documents()
    assert len(recs) >= 400
    assert any(r["label"] == "Article 21" for r in recs)
    assert any(r["label"] == "Preamble" for r in recs)


def test_loader_picks_up_bare_acts(tmp_path):
    (tmp_path / "bare_acts").mkdir()
    (tmp_path / "bare_acts" / "sample.json").write_text(json.dumps([{
        "document_name": "Sample Test Act", "provision_type": "Section", "provision_number": 1,
        "title": "Test title", "text": "Test text about widgets."}]), encoding="utf-8")
    recs = load_documents(tmp_path)
    assert recs[0]["label"] == "Section 1" and recs[0]["part"] == ""


def test_tokenize_keeps_clause_numbers():
    assert "19(1)(a)" in tokenize("Explain Article 19(1)(a).")


def test_analyze_removes_stopwords_and_stems():
    assert analyze("What are the Fundamental Rights?") == ["fundament", "right"]


# ---------------- regex ----------------
@pytest.mark.parametrize("text,label", [
    ("What is Article 21?", "Article 21"),
    ("Explain Art. 19(1)(a)", "Article 19(1)(a)"),
    ("article 14 says", "Article 14"),
    ("punishment under Section 302 IPC", "Section 302"),
    ("cheating u/s 420", "Section 420"),
    ("What about Article 21A?", "Article 21A"),
])
def test_regex_single(text, label):
    assert [r.label for r in find_references(text)] == [label]


def test_regex_multiple_and_part():
    labels = [r.label for r in find_references("Compare Articles 14, 15 and 16 of Part III")]
    assert labels == ["Article 14", "Article 15", "Article 16", "Part III"]


def test_regex_no_false_positive():
    assert find_references("Which right protects freedom of speech?") == []


def test_mask_and_clause():
    assert mask_references("What is Article 19(1)(a)?").split() == ["What", "is", "REF", "?"]
    text = "(1) All citizens shall have the right— (a) to freedom of speech and expression; (b) to assemble (2) Nothing"
    assert extract_clause(text, ["1", "a"]).startswith("(a) to freedom of speech")
    assert extract_clause(text, ["9"]) is None


# ---------------- HMM ----------------
@pytest.fixture(scope="session")
def tagger():
    return HMMPosTagger()


def test_hmm_tags_question(tagger):
    tags = dict(tagger.tag(["Which", "Article", "protects", "freedom", "of", "speech", "?"]))
    assert tags["Article"] == "NOUN" and tags["protects"] == "VERB" and tags["of"] == "ADP"


def test_hmm_unknown_word(tagger):
    assert signature("Zorblating") == "<-ing>" and signature("19(1)") == "<NUM>"
    tagged = tagger.tag(["what", "is", "zxqvon", "?"])
    assert len(tagged) == 4 and not tagger.is_known("zxqvon")


def test_hmm_empty(tagger):
    assert tagger.tag([]) == []


# ---------------- CYK ----------------
@pytest.mark.parametrize("q,qtype", [
    ("What is Article 21?", "LOOKUP"),
    ("Which Article deals with the Right to Equality?", "SEARCH"),
    ("Explain Article 19.", "EXPLAIN"),
    ("What are the fundamental rights?", "DEFINE"),
    ("Which Section deals with theft?", "SEARCH"),
    ("What does Article 32 provide?", "LOOKUP"),
])
def test_cyk_classification(qa, q, qtype):
    assert qa.ask(q).parse.question_type == qtype


def test_cyk_unparsable_falls_back(qa):
    a = qa.ask("rights speech the of is what")
    assert not a.parse.success and a.question_type == "SEARCH"


def test_cyk_table_and_tree(tagger):
    res = CYKParser().parse(tagger.tag(["Explain", "REF"]))
    assert "EXPLAIN" in res.table[0][1] and res.tree.label() == "EXPLAIN"


# ---------------- TF-IDF ----------------
def test_tfidf_without_number():
    r = TfidfRetriever(load_documents())
    assert r.search("freedom of speech and expression")[0].record["label"] == "Article 19"


def test_tfidf_explicit_boost():
    r = TfidfRetriever(load_documents())
    hit = r.search("Article 21", find_references("Article 21"))[0]
    assert hit.record["label"] == "Article 21" and hit.boost == 1.0


# ---------------- end-to-end (the 10 required cases) ----------------
@pytest.mark.parametrize("q,label", [
    ("What is Article 21?", "Article 21"),
    ("Which Article deals with equality before law?", "Article 14"),
    ("Explain Article 19(1)(a).", "Article 19"),
    ("Which Article protects freedom of speech?", "Article 19"),
    ("What does Article 32 provide?", "Article 32"),
    ("Which Article abolishes untouchability?", "Article 17"),
])
def test_e2e_found(qa, q, label):
    a = qa.ask(q)
    assert a.found and a.best.record["label"] == label


def test_e2e_clause(qa):
    assert "freedom of speech" in qa.ask("Explain Article 19(1)(a).").clause_text


def test_e2e_fundamental_rights(qa):
    a = qa.ask("What are the fundamental rights?")
    assert a.found and a.best.record["part"].startswith("Part III")


def test_e2e_missing_article(qa):
    a = qa.ask("What is Article 999?")
    assert not a.found and "not found" in a.answer


def test_e2e_irrelevant_question(qa):
    assert not qa.ask("How do I bake a chocolate cake with vanilla frosting?").found


def test_e2e_section_not_in_constitution(qa):
    assert not qa.ask("What is Section 302?", documents=["Constitution of India"]).found


@pytest.mark.parametrize("q", ["", "   ", "???", "a" * 5000, "Article", "12345", "<script>alert(1)</script>"])
def test_e2e_never_crashes(qa, q):
    qa.ask(q)
