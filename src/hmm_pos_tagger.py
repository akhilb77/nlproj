"""
hmm_pos_tagger.py
-----------------
Algorithm 2: Hidden Markov Model Part-of-Speech tagger.

  * Hidden states   = POS tags (Universal tagset: NOUN, VERB, ADJ, ...)
  * Observations    = words
  * Training        = supervised: count tag->tag transitions and tag->word
                      emissions in a tagged corpus (NLTK Penn Treebank sample
                      + a small hand-tagged set of legal questions), smoothed
                      with Lidstone (add-gamma) estimates.
  * Prediction      = Viterbi algorithm (most likely tag sequence).
  * Unknown words   = replaced by a "word shape" signature (e.g. <NUM>,
                      <CAP>, <-ing>, <-s>) learnt from rare training words.
"""

import nltk
from nltk.probability import LidstoneProbDist
from nltk.tag.hmm import HiddenMarkovModelTrainer

# Small hand-tagged corpus of question patterns used in this project's domain.
DOMAIN_CORPUS = """
what/PRON is/VERB article/NOUN 21/NUM ?/.
what/PRON are/VERB the/DET fundamental/ADJ rights/NOUN ?/.
which/DET article/NOUN deals/VERB with/ADP the/DET right/NOUN to/ADP equality/NOUN ?/.
which/DET article/NOUN protects/VERB freedom/NOUN of/ADP speech/NOUN ?/.
which/DET section/NOUN deals/VERB with/ADP theft/NOUN ?/.
explain/VERB article/NOUN 19/NUM ./.
what/PRON does/VERB article/NOUN 32/NUM provide/VERB ?/.
which/DET article/NOUN deals/VERB with/ADP equality/NOUN before/ADP law/NOUN ?/.
which/DET provision/NOUN guarantees/VERB protection/NOUN of/ADP life/NOUN and/CONJ personal/ADJ liberty/NOUN ?/.
what/PRON is/VERB the/DET punishment/NOUN for/ADP murder/NOUN ?/.
tell/VERB me/PRON about/ADP the/DET preamble/NOUN ./.
which/DET article/NOUN abolishes/VERB untouchability/NOUN ?/.
who/PRON elects/VERB the/DET president/NOUN ?/.
describe/VERB the/DET powers/NOUN of/ADP the/DET supreme/ADJ court/NOUN ./.
what/PRON is/VERB the/DET right/NOUN to/ADP education/NOUN ?/.
which/DET article/NOUN prohibits/VERB discrimination/NOUN on/ADP grounds/NOUN of/ADP religion/NOUN ?/.
what/PRON does/VERB section/NOUN 420/NUM say/VERB ?/.
which/DET article/NOUN covers/VERB emergency/NOUN provisions/NOUN ?/.
"""


def load_domain_corpus() -> list[list[tuple[str, str]]]:
    sents = []
    for line in DOMAIN_CORPUS.strip().splitlines():
        sents.append([tuple(tok.rsplit("/", 1)) for tok in line.split()])
    return sents


def load_treebank() -> list[list[tuple[str, str]]]:
    """Penn Treebank sample (~3900 sentences) mapped to the Universal tagset."""
    for res in ("treebank", "universal_tagset"):
        try:
            nltk.data.find(f"corpora/{res}" if res == "treebank" else f"taggers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    try:
        return list(nltk.corpus.treebank.tagged_sents(tagset="universal"))
    except LookupError:
        return []  # offline: train on the domain corpus only


def signature(word: str) -> str:
    """Word-shape class used for unknown / rare words."""
    if any(ch.isdigit() for ch in word):
        return "<NUM>"
    if not word.isalpha():
        return "<SYM>"
    low = word.lower()
    for suf in ("ing", "ed", "ly", "tion", "ity", "ment", "ness", "al", "ive", "ous", "able", "s"):
        if low.endswith(suf):
            return f"<-{suf}>"
    return "<CAP>" if word[0].isupper() else "<UNK>"


class HMMPosTagger:
    def __init__(self, rare_threshold: int = 1, domain_weight: int = 5):
        treebank = load_treebank()
        corpus = treebank + load_domain_corpus() * domain_weight
        self.corpus_size = len(corpus)
        self.used_treebank = bool(treebank)

        counts = nltk.FreqDist(w.lower() for s in corpus for w, _ in s)
        self.vocab = {w for w, c in counts.items() if c > rare_threshold}
        train = [[(self._obs(w), t) for w, t in s] for s in corpus]

        trainer = HiddenMarkovModelTrainer()
        self.model = trainer.train_supervised(
            train, estimator=lambda fd, bins: LidstoneProbDist(fd, 0.01, bins))

    def _obs(self, word: str) -> str:
        """Observation symbol: the lower-cased word if known, else its signature."""
        low = word.lower()
        return low if low in self.vocab else signature(word)

    def is_known(self, word: str) -> bool:
        return word.lower() in self.vocab

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        if not tokens:
            return []
        tags = self.model.best_path([self._obs(t) for t in tokens])  # Viterbi
        return list(zip(tokens, tags))


CONTENT_TAGS = {"NOUN", "VERB", "ADJ", "NUM", "X"}


def content_words(tagged: list[tuple[str, str]], stop_words: set[str]) -> list[str]:
    """Keep nouns / verbs / adjectives / numbers that are not stop words."""
    return [w for w, t in tagged if t in CONTENT_TAGS and w.lower() not in stop_words and w[0].isalnum()]
