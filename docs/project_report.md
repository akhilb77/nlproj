# NyayaQuery: An NLP-Based Question Answering System for the Constitution of India and Bare Acts

*Project report — Natural Language Processing course*

---

## 1. Abstract

Legal texts such as the Constitution of India are long, formally worded and hard for ordinary
citizens and students to search. NyayaQuery is a retrieval-based question answering system that
accepts a natural-language question and returns the original text of the most relevant Article
or Section together with its source and a relevance score. The system combines four classical NLP
techniques: regular expressions (finite state automata) to detect explicit legal references, a
Hidden Markov Model part-of-speech tagger to identify content words, the CYK parsing algorithm with
a context-free grammar in Chomsky Normal Form to analyse and classify the question, and TF-IDF
weighting with cosine similarity to rank provisions. Because answers are always extracted from the
source document, the system cannot fabricate legal content; when no provision is sufficiently
relevant it says so explicitly. The application is implemented in Python with NLTK, scikit-learn
and Streamlit and runs entirely offline.

## 2. Introduction

The Constitution of India contains a Preamble and over 450 Articles organised into Parts,
Chapters and headings. Students and citizens usually know *what* they are looking for
("freedom of speech") but not *where* it is (Article 19). Question answering (QA) systems bridge
this gap by mapping a question to the passage that answers it. This project builds such a system
using techniques taught in an NLP course so that each step is transparent and explainable.

## 3. Problem Statement

Given a question in English about the Constitution of India (or a Bare Act), retrieve the
provision that answers it and present its original text, number, title and source — with or
without an explicit Article number in the question — and clearly report when no relevant
provision exists, without generating unverified legal text.

## 4. Objectives

1. Convert the Constitution PDF into a structured, searchable dataset.
2. Detect explicit references (Article 21, Article 19(1)(a), Section 302) using regular expressions.
3. Tag questions with parts of speech using a trained Hidden Markov Model.
4. Parse and classify questions using the CYK algorithm.
5. Retrieve relevant provisions using TF-IDF and cosine similarity with reference boosting.
6. Integrate all components in one pipeline with a Streamlit user interface.
7. Provide an NLP analysis view that makes each algorithm's output visible.
8. Allow Bare Acts to be added without changing code.

## 5. Existing System

Today users rely on (a) reading the PDF and using the viewer's text search, which requires exact
words and has no ranking; (b) general web search engines, which return commentary of varying
reliability rather than the authoritative text; and (c) generative AI chatbots, which can
produce fluent but incorrect or fabricated provisions ("hallucination") and do not show the
exact source text.

## 6. Proposed System

NyayaQuery is a local, retrieval-based system over the official text. It understands explicit
references, ranks provisions by statistical relevance when no number is given, shows the exact
original text and its location (Part / heading), offers related provisions, and exposes its
reasoning through an NLP analysis panel. A similarity threshold prevents low-confidence answers.

## 7. Literature Review

- **Regular expressions and finite automata** — Kleene showed that regular expressions and finite
  automata describe the same class of languages; this is the basis of pattern matching in text
  processing (Jurafsky & Martin, ch. 2; Hopcroft, Motwani & Ullman).
- **Hidden Markov Models** — Rabiner's tutorial (1989) describes HMMs and the Viterbi algorithm;
  HMM taggers were the standard statistical POS taggers before discriminative and neural models
  (Jurafsky & Martin, ch. 8). Brants' TnT tagger (2000) showed HMM taggers with good unknown-word
  handling reach high accuracy.
- **CYK parsing** — The Cocke–Younger–Kasami algorithm (Younger 1967; Kasami 1965) recognises
  context-free languages in cubic time for grammars in Chomsky Normal Form (Chomsky 1959).
- **TF-IDF and the vector space model** — Salton, Wong & Yang (1975) introduced the vector space
  model; Spärck Jones (1972) introduced inverse document frequency. Manning, Raghavan & Schütze
  (2008) cover TF-IDF ranking and cosine similarity.
- **Legal information retrieval** — Retrieval over statutes is a long-standing research topic
  (e.g. COLIEE competition tasks on statute law retrieval), where lexical methods such as TF-IDF
  and BM25 remain strong baselines.

## 8. System Architecture

See `docs/architecture.md` for the Mermaid diagrams. The layers are:

- **Data layer** – JSON datasets in `data/` and `data/bare_acts/`, loaded by `document_loader.py`.
- **NLP layer** – `text_preprocessor`, `regex_analyzer`, `hmm_pos_tagger`, `cyk_parser`.
- **Retrieval layer** – `information_retrieval.py` (TF-IDF index).
- **Pipeline** – `question_answering.py` (`NyayaQuery.ask`).
- **Presentation** – `app.py` (Streamlit).

## 9. Methodology

1. **Dataset creation** – The Constitution PDF was converted to text with `pdftotext`, the table of
   contents was parsed to obtain the authoritative list of Articles, page headers/footnotes/
   amendment markers were removed, and the body was split per Article (`pdf_to_dataset.py`).
2. **Normalisation** – Unicode normalisation, quote/dash unification, whitespace collapse.
3. **Reference detection** – regex patterns produce structured `LegalReference` objects.
4. **POS tagging** – HMM trained on a tagged corpus; Viterbi decoding.
5. **Parsing** – references are masked as `REF`; CYK parses the tagged tokens; the start symbol
   gives the question type.
6. **Retrieval** – TF-IDF over provision texts; cosine similarity with the query; explicit
   reference boost (+1.0) and Part boost (+0.15).
7. **Decision** – non-existent references → "not found"; best score below threshold → "no
   sufficiently relevant provision"; otherwise answer with source, clause text and related provisions.

## 10. Algorithms

Summarised here; full explanation with examples and complexity in `docs/algorithms.md`.

| Algorithm | Input | Output | Complexity |
|-----------|-------|--------|-----------|
| Regex / FSA | question string | list of references | O(n) |
| HMM + Viterbi | tokens | POS tags | O(n·T²) |
| CYK | tagged tokens | table, tree, question type | O(n³·\|G\|) |
| TF-IDF + cosine | query string | ranked provisions | O(non-zeros in matrix) |

## 11. Implementation

- **Language / libraries**: Python 3.11, NLTK (HMM trainer, Porter stemmer, Treebank corpus,
  `Tree`), scikit-learn (`TfidfVectorizer`, `cosine_similarity`), NumPy, pandas, Streamlit, pytest.
- **HMM**: `nltk.tag.hmm.HiddenMarkovModelTrainer.train_supervised` with Lidstone (γ = 0.01)
  estimates; word-shape signatures for rare/unknown words.
- **CYK**: implemented from scratch in `cyk_parser.py` (table filling, back-pointers, tree building);
  grammar of 24 binary rules plus a lexicon.
- **IR**: custom analyzer (tokenise → stop-word removal → stemming), `sublinear_tf=True`,
  titles counted twice.
- **UI**: sidebar with document selector, top-k and threshold sliders; example question buttons;
  answer panel; related provisions; NLP analysis expander with tables for each algorithm.
- **Testing**: `tests/test_nyayaquery.py` covers every module and end-to-end cases, including
  invalid inputs.

## 12. Dataset

| Property | Value |
|----------|-------|
| Source | Constitution of India (official PDF) |
| Records | 465 (1 Preamble + 464 Articles) |
| Fields | document_name, provision_type, provision_number, title, part, heading, text |
| Files | `data/constitution_of_india.json`, `data/constitution_of_india.txt` |
| Excluded | Schedules, footnotes/amendment history, omitted/repealed Articles |

Bare Acts can be added as JSON files with the same schema in `data/bare_acts/`; none are bundled.

## 13. Results and Discussion

### 13.1 Functional verification

The automated test suite (`python -m pytest -v`) checks, among others, that:
"What is Article 21?" → Article 21; "Which Article deals with equality before law?" → Article 14;
"Which Article protects freedom of speech?" → Article 19; "Explain Article 19(1)(a)." → Article 19
with clause (1)(a) extracted; "What are the fundamental rights?" → a Part III provision;
"What is Article 999?" → not found; an unrelated cooking question → no relevant provision; and
the CYK parser classifies the six example question forms correctly. The number of passing tests
is reported by pytest when run.

### 13.2 Evaluation (to be completed by the student)

> **No quantitative evaluation has been performed yet. Do not report numbers that were not measured.**

Suggested procedure:

1. Write 50–100 test questions, about half naming an Article and half not, and record the correct
   Article(s) for each in a CSV file (`question, expected_article`).
2. Run every question through `NyayaQuery().ask(q)` and record the top-5 labels.
3. Compute **Top-1 accuracy** (best answer correct), **Top-5 recall** (correct Article in top 5)
   and **MRR** (mean of 1/rank of the correct Article).
4. Report results separately for questions with and without explicit references, and compare
   with a baseline (TF-IDF without HMM content-word weighting, or without the boost).
5. For the HMM, evaluate tagging accuracy on a held-out 10% of the Treebank sample
   (`tagger.model.accuracy(test_sents)` after training on the other 90%).
6. For CYK, report the percentage of test questions the grammar parses.

| Metric | Value |
|--------|-------|
| Top-1 accuracy | *to be measured* |
| Top-5 recall | *to be measured* |
| MRR | *to be measured* |
| HMM tagging accuracy (held-out) | *to be measured* |
| CYK parse coverage | *to be measured* |

### 13.3 Discussion

Explicit references are resolved reliably because they are matched exactly and boosted. Questions
without numbers work well when the user's words appear in the Article title or text (e.g. "freedom
of speech", "untouchability"). Questions phrased with synonyms not present in the text are the main
source of errors, which is an inherent limitation of lexical retrieval.

## 14. Limitations

- Lexical retrieval: no synonym or semantic matching.
- Grammar covers common question shapes only; others fall back to SEARCH.
- HMM trained mainly on newswire; legal vocabulary may be mis-tagged.
- Only the Constitution is bundled; Schedules and amendment footnotes are not included.
- Some extraction artefacts may remain in Article texts.
- Retrieves text only; no legal interpretation or advice.

## 15. Future Scope

- Add the Bharatiya Nyaya Sanhita / IPC, CrPC/BNSS and other Bare Acts.
- BM25 ranking, query expansion with a legal thesaurus, or sentence embeddings.
- Larger hand-annotated legal POS corpus; probabilistic CFG (PCFG) for ranking parses.
- Answer highlighting at sentence level; multilingual (Hindi) support; links to case law.

## 16. Conclusion

NyayaQuery shows that a transparent pipeline of classical NLP algorithms can answer questions
about the Constitution of India with exact, verifiable source text. Regular expressions handle
explicit references, the HMM identifies important words, the CYK parser reveals question
structure, and TF-IDF with cosine similarity finds relevant provisions even without an Article
number. Because nothing is generated, the system avoids fabricating legal content.

## 17. References

1. D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. draft. https://web.stanford.edu/~jurafsky/slp3/
2. C. D. Manning, P. Raghavan and H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008.
3. L. R. Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition," *Proceedings of the IEEE*, 77(2), 1989.
4. T. Brants, "TnT – A Statistical Part-of-Speech Tagger," *Proc. ANLP*, 2000.
5. D. H. Younger, "Recognition and parsing of context-free languages in time n³," *Information and Control*, 10(2), 1967.
6. G. Salton, A. Wong and C. S. Yang, "A Vector Space Model for Automatic Indexing," *Communications of the ACM*, 18(11), 1975.
7. K. Spärck Jones, "A statistical interpretation of term specificity and its application in retrieval," *Journal of Documentation*, 28(1), 1972.
8. J. E. Hopcroft, R. Motwani and J. D. Ullman, *Introduction to Automata Theory, Languages, and Computation*, Pearson.
9. S. Bird, E. Klein and E. Loper, *Natural Language Processing with Python*, O'Reilly, 2009. https://www.nltk.org/book/
10. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 12, 2011.
11. Legislative Department, Government of India, *The Constitution of India*. https://legislative.gov.in/constitution-of-india/
