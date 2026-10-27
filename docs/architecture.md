# NyayaQuery – Architecture

## System architecture

```mermaid
flowchart TD
    U[User question - Streamlit app.py] --> N[text_preprocessor.normalize]
    N --> R[regex_analyzer.find_references<br/>Algorithm 1: Regex/FSA]
    N --> T[tokenize]
    T --> H[hmm_pos_tagger.HMMPosTagger.tag<br/>Algorithm 2: HMM + Viterbi]
    R --> M[mask_references -> REF token]
    M --> H2[HMM tags of masked question]
    H2 --> C[cyk_parser.CYKParser.parse<br/>Algorithm 3: CYK]
    H --> K[content words]
    K --> Q[retrieval query]
    R --> I
    Q --> I[information_retrieval.TfidfRetriever.search<br/>Algorithm 4: TF-IDF + cosine + boost]
    D[(data/*.json<br/>data/bare_acts/*.json)] --> L[document_loader.load_documents] --> I
    I --> V{reference exists?<br/>score >= threshold?}
    C --> A
    V -- yes --> A[Answer: provision, title, original text,<br/>source, score, related, NLP analysis]
    V -- no --> X[No sufficiently relevant provision / Article not found]
```

## Data flow

1. **Load** – `document_loader` reads every JSON file once at start-up and adds `id` and `label`.
2. **Index** – `TfidfRetriever` fits the TF-IDF matrix over all provisions; `HMMPosTagger` trains;
   both are cached by Streamlit (`st.cache_resource`), so this happens once per server.
3. **Query** – `NyayaQuery.ask()` runs the pipeline and returns an `Answer` dataclass.
4. **Display** – `app.py` renders the answer, related provisions and the NLP analysis panel.

## Modules

| Module | Responsibility |
|--------|----------------|
| `pdf_to_dataset.py` | One-time conversion of the Constitution PDF into JSON (existing, unchanged) |
| `document_loader.py` | Load and validate JSON datasets; Bare Acts are auto-discovered |
| `text_preprocessor.py` | Normalisation, tokenisation, stop words, Porter stemming |
| `regex_analyzer.py` | Reference detection, masking, clause extraction |
| `hmm_pos_tagger.py` | HMM training, Viterbi tagging, unknown-word signatures, content words |
| `cyk_parser.py` | CNF grammar, CYK table, parse tree, question type |
| `information_retrieval.py` | TF-IDF vectoriser, cosine similarity, boosting, document filter |
| `question_answering.py` | Orchestrates everything; confidence check; no-answer handling |
| `app.py` | Streamlit UI |

## Question answering workflow

```mermaid
sequenceDiagram
    participant UI as Streamlit
    participant QA as NyayaQuery.ask
    participant RX as Regex
    participant HMM
    participant CYK
    participant IR as TF-IDF
    UI->>QA: question, selected documents, top_k, threshold
    QA->>RX: find_references(question)
    RX-->>QA: [Article 19 (1)(a)]
    QA->>HMM: tag(tokens)
    HMM-->>QA: [(Explain, VERB), ...] + content words
    QA->>CYK: parse(tag(masked tokens))
    CYK-->>QA: EXPLAIN + tree + table
    QA->>IR: search(question + content words, refs)
    IR-->>QA: ranked hits (cosine + boost)
    QA->>QA: reference exists? best score >= threshold? extract clause
    QA-->>UI: Answer
```

## Extending

- **New Bare Act** – add `data/bare_acts/<name>.json` with the same schema.
- **New question shape** – add CNF rules to `GRAMMAR` and words to `LEXICON` in `cyk_parser.py`.
- **Better tagging** – add hand-tagged sentences to `DOMAIN_CORPUS` in `hmm_pos_tagger.py`.
