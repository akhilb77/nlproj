# ⚖️ NyayaQuery

**An NLP-based Question Answering System for the Constitution of India and Bare Acts.**

Ask a question in plain English ("Which Article protects freedom of speech?") and NyayaQuery
returns the **original legal text** of the most relevant Article/Section, with its source and a
similarity score. It never generates or invents legal text — every answer is a provision copied
from the dataset, or an explicit "not found".

## Features

- Natural-language questions with or without Article numbers
- Explicit reference detection: `Article 21`, `Art. 19(1)(a)`, `Articles 14 and 21`, `Section 302`, `u/s 420`, `Part III`
- Clause extraction: `Article 19(1)(a)` shows the full Article and highlights clause (1)(a)
- Related provisions
- NLP analysis panel showing the output of all four algorithms
- Document selector (Constitution / individual Bare Acts / all)
- Runs fully offline after the first NLTK download; no paid APIs, no LLMs

## Algorithms

| # | Algorithm | File | Role |
|---|-----------|------|------|
| 1 | Regular Expressions / FSA | `src/regex_analyzer.py` | Detect explicit Article/Section/Part references |
| 2 | Hidden Markov Model (Viterbi) | `src/hmm_pos_tagger.py` | POS-tag the question, pick content words |
| 3 | CYK parsing (CNF grammar) | `src/cyk_parser.py` | Parse question structure, classify LOOKUP/DEFINE/EXPLAIN/SEARCH |
| 4 | TF-IDF + cosine similarity | `src/information_retrieval.py` | Rank provisions, boost explicit references |

Details: [docs/algorithms.md](docs/algorithms.md) · Architecture: [docs/architecture.md](docs/architecture.md)

## Project structure

```
nyayaquery/
├── app.py                         Streamlit UI
├── requirements.txt
├── pytest.ini
├── data/
│   ├── constitution_of_india.json 465 records (Preamble + 464 Articles)
│   ├── constitution_of_india.txt  same data, human-readable
│   └── bare_acts/                 drop extra Act JSON files here
├── src/
│   ├── pdf_to_dataset.py          (existing) PDF -> JSON converter
│   ├── document_loader.py         loads all JSON datasets
│   ├── text_preprocessor.py       normalise, tokenise, stop words, stemming
│   ├── regex_analyzer.py          Algorithm 1
│   ├── hmm_pos_tagger.py          Algorithm 2
│   ├── cyk_parser.py              Algorithm 3
│   ├── information_retrieval.py   Algorithm 4
│   └── question_answering.py      full pipeline
├── tests/test_nyayaquery.py
└── docs/  algorithms.md, architecture.md, project_report.md, viva_questions.md
```

## Installation (Windows PowerShell)

Use Python **3.10, 3.11 or 3.12** (pre-release Pythons such as 3.15 alpha may not have scikit-learn wheels).

```powershell
cd nyayaquery
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('treebank'); nltk.download('universal_tagset')"
```

If `activate` is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

The NLTK download (~10 MB) is the Penn Treebank sample used to train the HMM. The code also
tries to download it automatically; if you are offline the HMM trains only on the built-in
domain corpus (still works, less accurate).

## Run

```powershell
streamlit run app.py
```

Opens http://localhost:8501. The first load takes a few seconds (HMM training + TF-IDF index).

## Tests

```powershell
python -m pytest -v
```

## Dataset setup

`data/constitution_of_india.json` is already included. Schema of each record:

```json
{"document_name": "Constitution of India", "provision_type": "Article", "provision_number": "21",
 "title": "Protection of life and personal liberty", "part": "Part III - Fundamental Rights",
 "heading": "Right to Freedom", "text": "No person shall be deprived of ..."}
```

**Adding a Bare Act:** save a JSON list with the same fields (e.g. `provision_type: "Section"`,
`document_name: "Indian Penal Code, 1860"`) as `data/bare_acts/ipc.json`. Restart the app — it
appears in the document selector and `Section 302` queries will resolve. No code changes.

To rebuild the Constitution dataset from the PDF (needs poppler's `pdftotext` on PATH):
`python -m src.pdf_to_dataset path\to\constitution.pdf`

## Example questions

- What is Article 21?
- Which Article deals with equality before law?
- Explain Article 19(1)(a).
- Which Article protects freedom of speech?
- What are the fundamental rights?
- What does Article 32 provide?
- Which Article abolishes untouchability?
- Tell me about the Preamble

## How to verify each algorithm works

Open the **NLP analysis** expander under any answer:

1. **Regex** – ask `Explain Art. 19(1)(a)`: table shows `Article 19(1)(a)`, clauses `(1)(a)`.
2. **HMM** – the POS table tags `protects` as VERB, `Article` as NOUN; nonsense words are marked "known word = False" and still get a tag via their word-shape.
3. **CYK** – a parse tree (e.g. `(SEARCH (WHICHNP ...) (VP ...))`) and the filled CYK table cells. Ask `rights speech the of is what` to see parse failure + fallback.
4. **TF-IDF** – the scores table shows cosine, boost (1.0 for explicit references) and final score. Ask a question without a number to see pure cosine ranking.

`python -m pytest -v` checks all four automatically.

## Known limitations

- Only the Constitution is bundled; no Bare Act (IPC etc.) data is included, so `Section 302` returns "not found" until you add one.
- TF-IDF is lexical: questions using words absent from the text (e.g. "free speech" works, "right to protest" may rank poorly) can miss. No synonyms/embeddings.
- The CYK grammar covers common question shapes only; other shapes fall back to SEARCH (retrieval still works).
- The HMM is trained mostly on newswire (Penn Treebank), so some legal words are mis-tagged.
- Footnotes/amendment history were removed during dataset extraction; a few Articles may carry small extraction artefacts.
- It retrieves provisions; it does not interpret law or give legal advice.
