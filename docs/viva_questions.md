# Viva Questions and Answers – NyayaQuery

## Project overview

**1. What is NyayaQuery?**
A question answering system for the Constitution of India (and Bare Acts added later). The user
asks a question in English and it returns the original text of the most relevant Article/Section.

**2. Is it a generative system like ChatGPT?**
No. It is *extractive/retrieval-based*: every answer is a provision copied from the dataset. It
cannot hallucinate a law; if nothing relevant is found it says so.

**3. Which four algorithms does it use?**
Regular expressions (FSA) for references, HMM for POS tagging, CYK for parsing/question
classification, and TF-IDF with cosine similarity for retrieval.

**4. What does the dataset contain?**
465 records: the Preamble and 464 Articles, each with number, title, Part, heading and text,
extracted from the official PDF by `src/pdf_to_dataset.py`.

**5. Why not just use keyword search (Ctrl+F)?**
Ctrl+F needs exact words and gives no ranking. TF-IDF ranks by importance of shared terms, stemming
matches "protects"/"protection", and explicit references, parsing and tagging add structure.

## NLP fundamentals

**6. What is tokenisation?**
Splitting text into units (tokens). Our regex tokenizer keeps `19(1)(a)` as one token.

**7. What are stop words and why remove them?**
Very frequent words (the, is, which) that carry little meaning. Removing them stops them from
dominating similarity scores.

**8. What is stemming? Which stemmer do you use?**
Reducing words to a root form by chopping suffixes: "protection", "protects" → "protect". We use
NLTK's Porter stemmer.

**9. Stemming vs lemmatisation?**
Stemming uses rules and may produce non-words ("fundament"); lemmatisation uses a dictionary and
returns real words ("better"→"good"). Stemming is faster and needs no extra data.

## Regular expressions

**10. What is a regular expression?**
A pattern describing a set of strings, e.g. `\d{1,3}` = one to three digits.

**11. How are regex and FSA related?**
Every regular expression can be converted to an equivalent finite state automaton and vice versa
(Kleene's theorem). Matching a regex = running the FSA over the input.

**12. What references can your regex detect?**
`Article 21`, `Art. 19(1)(a)`, `article 14`, `Articles 14, 15 and 16`, `Section 302`, `Sec. 420`,
`u/s 420`, `Article 21A`, `Part III`, `Preamble`.

**13. What happens after a reference is detected?**
The provision gets a +1.0 score boost; the reference is replaced by `REF` for the parser; if it
does not exist (Article 999) we say "not found"; clause numbers are used to extract the clause.

**14. Limitation of regex?**
It cannot handle nested/recursive structure (e.g. matching balanced brackets) and only finds what
the pattern anticipates — "the twenty-first Article" is not detected.

## HMM

**15. What is a Hidden Markov Model?**
A probabilistic model where a sequence of hidden states (POS tags) generates a sequence of
observations (words). Defined by initial, transition and emission probabilities.

**16. What is the Markov assumption?**
The current tag depends only on the previous tag (first-order), and each word depends only on its own tag.

**17. How is your HMM trained?**
Supervised training with NLTK's `HiddenMarkovModelTrainer` on the Penn Treebank sample (Universal
tagset) plus a small hand-tagged legal question set. It counts transitions and emissions and
applies Lidstone smoothing.

**18. How does it predict tags?**
With the Viterbi algorithm — dynamic programming that finds the single most likely tag sequence
in O(n·T²).

**19. How do you handle unknown words?**
Rare training words are replaced by shape signatures (`<NUM>`, `<-ing>`, `<-tion>`, `<CAP>`),
so the model learns tag probabilities for each shape; unknown words at test time use the same
signature. Smoothing prevents zero probabilities.

**20. Why use POS tags in a QA system?**
To pick content words (nouns, verbs, adjectives, numbers) which are given extra weight in the
retrieval query, and to provide pre-terminals for the CYK parser.

## CYK parsing

**21. What is a context-free grammar?**
A set of rules `A → α` where the left side is one non-terminal. It can describe nested structure
like phrases inside phrases.

**22. What is Chomsky Normal Form?**
A CFG where every rule is `A → B C` or `A → terminal`. CYK requires it.

**23. Explain the CYK algorithm.**
Bottom-up dynamic programming. Fill a triangular table where cell [i][j] holds every non-terminal
that can generate words i..j: diagonal from the lexicon, then longer spans by combining two
smaller adjacent spans with a binary rule. Sentence is accepted if the start symbol is in [0][n-1].

**24. Time complexity of CYK?**
O(n³·|G|) time and O(n²) space.

**25. How do you classify questions?**
The start symbol that spans the whole question is the type: LOOKUP (what is Article 21),
DEFINE (what are the fundamental rights), EXPLAIN (explain Article 19), SEARCH (which Article
deals with theft).

**26. What if the question does not parse?**
We fall back to LOOKUP if a reference was found, otherwise SEARCH. Retrieval still runs, so the
user still gets an answer.

## TF-IDF, cosine similarity, IR

**27. What is TF-IDF?**
Term Frequency × Inverse Document Frequency. A word scores high in a document if it is frequent
there but rare across the collection.

**28. Why IDF?**
Words like "State" or "Parliament" appear in hundreds of Articles; IDF lowers their weight so rarer,
more specific words decide the ranking.

**29. What is cosine similarity and why not Euclidean distance?**
cos θ = q·d / (|q||d|). It measures direction, not length, so long Articles are not penalised or
favoured just for their length.

**30. How does the system answer without an Article number?**
The question is turned into a TF-IDF vector and compared with all 465 provision vectors; the
highest cosine similarity wins (e.g. "freedom of speech" → Article 19).

**31. What is the similarity threshold for?**
If the best score is below it (default 0.08) the system says no sufficiently relevant provision
was found instead of returning a random Article.

**32. What are precision and recall?**
Precision = relevant results retrieved / all retrieved. Recall = relevant retrieved / all relevant.
Our top-1 accuracy on a labelled question set can be measured as described in the report.

## Architecture

**33. Describe the pipeline.**
Normalise → regex references → HMM tags → CYK parse → TF-IDF retrieval → reference boost →
threshold check → answer with source.

**34. How do you add a new Bare Act?**
Put a JSON file with the same schema in `data/bare_acts/`. The loader picks it up automatically.

**35. Why Streamlit?**
Pure Python, no HTML/JS needed, quick to build an interactive demo; `st.cache_resource` keeps the
trained models in memory.

## Limitations and future scope

**36. Main limitations?**
Lexical matching only (no synonyms), limited grammar coverage, HMM trained on newswire, only the
Constitution bundled, no legal interpretation.

**37. Future scope?**
Add IPC/BNS and other Acts, BM25 or dense embeddings, synonym expansion with a legal thesaurus,
Hindi/regional language support, case-law linking, a larger hand-tagged legal POS corpus.
