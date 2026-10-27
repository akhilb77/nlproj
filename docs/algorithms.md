# Algorithms used in NyayaQuery

NyayaQuery uses four classic NLP algorithms, one after another. Each one adds a different kind
of information about the user's question.

---

## 1. Regular Expressions / Finite State Automata

**Definition.** A regular expression (regex) is a pattern that describes a set of strings. Every
regex is equivalent to a Finite State Automaton (FSA): a machine with a finite number of states
that reads characters one by one and accepts the string if it ends in an accepting state.

**How it works here** (`src/regex_analyzer.py`). The main pattern is

```
(articles?|arts?|sections?|secs?|u/s|s) \.? \s*  NUMBER CLAUSES ( (,|and|or|&|to) NUMBER CLAUSES )*
NUMBER  = \d{1,3}[A-Z]{0,2}          21, 21A, 243ZH
CLAUSES = (\(\w{1,4}\))*            (1)(a)
```

The FSA moves: *start → keyword → optional dot → number → (clause)\* → (separator → number)\* → accept*.
Separate patterns detect `Part III` and `Preamble`. Each match becomes a `LegalReference`
object `{type, number, clauses, raw text, span}`.

**Why.** Explicit references are the strongest possible signal. If the user writes "Article 21",
we know exactly which record to return — no statistics needed.

**Input / output.** `"Explain Art. 19(1)(a) and 21"` → `[Article 19 clauses [1,a], Article 21]`.

**Complexity.** O(n) in the length of the question (an FSA reads each character once; Python's
backtracking engine is linear for these patterns).

**Role in pipeline.** (a) explicit references get a +1.0 retrieval boost, (b) references are
replaced by a single `REF` token before CYK parsing, (c) missing references ("Article 999")
produce an honest "not found" message, (d) clause numbers are used to extract the clause text.

---

## 2. Hidden Markov Model (HMM) POS tagging

**Definition.** A HMM is a probabilistic model with *hidden states* that we cannot see and
*observations* that we can see. For POS tagging the hidden states are tags (NOUN, VERB, ADJ,
ADP, DET, ...) and the observations are words.

It has three parameter sets:
- **Initial probabilities** π(t) — probability a sentence starts with tag t
- **Transition probabilities** A(t₁→t₂) — probability tag t₂ follows tag t₁
- **Emission probabilities** B(t→w) — probability tag t produces word w

**Training** (`HMMPosTagger.__init__`). Supervised: NLTK's `HiddenMarkovModelTrainer.train_supervised`
counts these events in a tagged corpus — the NLTK Penn Treebank sample (~3,900 sentences,
mapped to the 12-tag Universal tagset) plus 18 hand-tagged legal questions (repeated 5× to give
domain patterns weight). Counts are turned into probabilities with Lidstone smoothing
(add 0.01) so unseen events never have probability zero.

**Unknown words.** Words seen at most once in training are replaced by a *word-shape signature*
(`<NUM>`, `<-ing>`, `<-tion>`, `<-s>`, `<CAP>`, `<UNK>` ...). The HMM therefore learns e.g.
"words ending in -tion are usually NOUN". At prediction time any unknown word is mapped to
its signature too, so it still gets a sensible tag.

**Prediction.** The Viterbi algorithm (dynamic programming) finds the tag sequence with the
highest probability:
`V[t][i] = max over t' ( V[t'][i-1] · A(t'→t) ) · B(t→wᵢ)`, then follows back-pointers.

**Why.** Tags tell us which words matter. Nouns, verbs, adjectives and numbers become
*content words*, which are added to the retrieval query a second time (double weight).
Tags also give the CYK parser its pre-terminals for open-class words.

**Example.** `Which Article protects freedom of speech ?` →
`DET NOUN VERB NOUN ADP NOUN .` → content words `freedom, speech, protects`.

**Complexity.** Viterbi is O(n · T²) for n words and T tags (T = 12). Training is O(corpus size).

---

## 3. CYK parsing

**Definition.** The Cocke–Younger–Kasami algorithm decides whether a sentence can be generated
by a context-free grammar (CFG) in Chomsky Normal Form, and builds its parse tree. In CNF every
rule is either `A → B C` (two non-terminals) or `A → word`.

**Grammar** (`src/cyk_parser.py`, `GRAMMAR`):

```
LOOKUP  → WHAT BEREF      BEREF  → BE REFNP          what is Article 21
LOOKUP  → WHAT DONP       DONP   → DO NPVP           what does Article 32 provide
                          NPVP   → NP VP
DEFINE  → WHAT BENP       BENP   → BE NP             what are the fundamental rights
EXPLAIN → EXPL NP                                    explain Article 19
EXPLAIN → EXPL MEPP       MEPP   → ME PP             tell me about the Preamble
SEARCH  → WHICHNP VP      WHICHNP→ WHICH NP          which Article deals with theft
VP → V PP | V NP | VP PP | BE PP
NP → DT NP | ADJ NP | N NP | NP PP | NP CONJNP       CONJNP → CONJ NP
PP → P NP                 REFNP  → DT REFNP
```

Pre-terminals come from a small closed-class lexicon (`what, is, the, with, explain` ...) and,
for every other word, from the HMM tag (NOUN→N, VERB→V, ADJ→ADJ ...). Unit rules such as
`NP → N` are folded into the lexicon (a noun cell holds both `N` and `NP`) so the grammar stays in CNF.
Explicit references were replaced by the token `REF` (which yields `REF, NP, REFNP`).

**How it works.** Build an n×n triangular table. `table[i][i]` = pre-terminals of word i.
Then for each span length 2..n, each start i, each split k:
if `B ∈ table[i][k]`, `C ∈ table[k+1][j]` and `A → B C` exists, add A to `table[i][j]` and
remember `(k, B, C)` as a back-pointer. If a start symbol (LOOKUP/EXPLAIN/SEARCH/DEFINE) is in
`table[0][n-1]` the sentence is accepted; the back-pointers rebuild the tree.

**Example.** `Explain REF` → table[0][0]={EXPL}, table[1][1]={REF,NP,REFNP}, table[0][1]={EXPLAIN}
→ tree `(EXPLAIN (EXPL Explain) (NP REF))` → type EXPLAIN.

**Why.** Shows the *structure* of the question and gives a question type, displayed in the UI
and returned with the answer. If parsing fails, the system falls back to LOOKUP (if a reference
was found) or SEARCH — retrieval runs either way, so the app never breaks.

**Complexity.** O(n³ · |G|) time, O(n²) space (n = words, |G| = grammar size).

---

## 4. TF-IDF and cosine similarity

**Definition.** TF-IDF weighs a term by how often it appears in a document (Term Frequency)
times how rare it is across all documents (Inverse Document Frequency):

`tfidf(t,d) = (1 + log tf(t,d)) · (log((1+N)/(1+df(t))) + 1)` (scikit-learn, `sublinear_tf=True`).

Cosine similarity measures the angle between two vectors:
`cos(q,d) = (q·d) / (|q|·|d|)` — 1 means same direction, 0 means no shared terms.

**How it works here** (`src/information_retrieval.py`).
1. Each provision → text = label + title (twice, so titles weigh more) + Part + heading + legal text.
2. Text is processed by `analyze()`: tokenise, lower-case, remove stop words, Porter-stem.
3. `TfidfVectorizer` builds a 465 × V sparse matrix (L2-normalised).
4. The query (question + HMM content words) is transformed the same way.
5. Cosine similarity with every provision; explicitly referenced provisions get **+1.0**,
   Articles in a referenced Part get +0.15.
6. Sort, keep top-k, and if the best score is below the threshold (default 0.08) report
   "no sufficiently relevant provision".

**Example.** "Which Article protects freedom of speech?" → stems `protect, freedom, speech` →
highest cosine is Article 19 ("Protection of certain rights regarding freedom of speech, etc").

**Complexity.** Indexing O(total tokens). Each query O(nnz of matrix) ≈ linear in corpus size.

**Role.** This is what finds the answer when no Article number is given, and ranks related provisions.
