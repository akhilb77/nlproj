"""
cyk_parser.py
-------------
Algorithm 3: CYK (Cocke-Younger-Kasami) parsing with a CFG in
Chomsky Normal Form (every rule is  A -> B C  or  A -> terminal).

Input : list of (word, POS) pairs, where explicit references such as
        "Article 21" have already been replaced by the single token REF.
Output: parse table, parse tree, and a question type:
        LOOKUP / DEFINE / EXPLAIN / SEARCH  (or None if parsing fails).
"""

from dataclasses import dataclass, field

from nltk import Tree

# ---------------------------------------------------------------------------
# Lexicon: closed-class words -> pre-terminal.  Open-class words use HMM tags.
# ---------------------------------------------------------------------------
LEXICON = {
    "WHAT": {"what"}, "WHICH": {"which"},
    "BE": {"is", "are", "was", "were"},
    "DO": {"does", "do", "did"},
    "EXPL": {"explain", "describe", "define", "elaborate", "summarize", "summarise", "list", "show", "tell"},
    "ME": {"me", "us"},
    "DT": {"the", "a", "an", "all", "any", "this", "that", "these", "those"},
    "P": {"with", "about", "of", "for", "to", "in", "on", "under", "against", "regarding", "before",
          "by", "from", "into", "between", "within", "among"},
    "CONJ": {"and", "or"},
    "V": {"deal", "deals", "protect", "protects", "provide", "provides", "cover", "covers", "guarantee",
          "guarantees", "say", "says", "state", "states", "mean", "means", "contain", "contains", "talk",
          "talks", "relate", "relates", "prohibit", "prohibits", "define", "defines", "abolish", "abolishes",
          "punish", "punishes", "allow", "allows", "give", "gives", "grant", "grants", "ensure", "ensures"},
}
HMM_TO_PRETERMINAL = {"NOUN": "N", "NUM": "N", "X": "N", "PRON": "N", "VERB": "V", "ADJ": "ADJ",
                      "ADV": "ADJ", "ADP": "P", "DET": "DT", "CONJ": "CONJ", "PRT": "P"}

# A pre-terminal also stands for the phrase it can start alone (unit rules folded into the lexicon).
LEXICAL_EXPANSION = {"N": {"N", "NP"}, "V": {"V", "VP"}, "REF": {"REF", "NP", "REFNP"}}

# ---------------------------------------------------------------------------
# Grammar in CNF.  Each entry:  (LHS, RHS1, RHS2)
# ---------------------------------------------------------------------------
GRAMMAR = [
    # question types
    ("LOOKUP", "WHAT", "BEREF"),     # what is Article 21
    ("BEREF", "BE", "REFNP"),
    ("LOOKUP", "WHAT", "DONP"),      # what does Article 32 provide
    ("DONP", "DO", "NPVP"),
    ("NPVP", "NP", "VP"),
    ("DEFINE", "WHAT", "BENP"),      # what are the fundamental rights
    ("BENP", "BE", "NP"),
    ("EXPLAIN", "EXPL", "NP"),       # explain Article 19
    ("EXPLAIN", "EXPL", "MEPP"),     # tell me about the preamble
    ("MEPP", "ME", "PP"),
    ("SEARCH", "WHICHNP", "VP"),     # which Article deals with theft
    ("WHICHNP", "WHICH", "NP"),
    # verb phrases
    ("VP", "V", "PP"), ("VP", "V", "NP"), ("VP", "VP", "PP"), ("VP", "BE", "PP"),
    # noun phrases
    ("NP", "DT", "NP"), ("NP", "ADJ", "NP"), ("NP", "N", "NP"), ("NP", "NP", "PP"),
    ("NP", "NP", "CONJNP"), ("CONJNP", "CONJ", "NP"),
    ("REFNP", "DT", "REFNP"),
    ("PP", "P", "NP"),
]
START_SYMBOLS = ["LOOKUP", "EXPLAIN", "SEARCH", "DEFINE"]  # priority order when ambiguous


@dataclass
class ParseResult:
    tokens: list[str]
    preterminals: list[set[str]]
    table: list[list[set[str]]] = field(default_factory=list)
    question_type: str | None = None
    tree: Tree | None = None

    @property
    def success(self) -> bool:
        return self.question_type is not None


def preterminals_for(word: str, pos: str) -> set[str]:
    if word == "REF":
        base = {"REF"}
    else:
        low = word.lower()
        base = {pt for pt, words in LEXICON.items() if low in words}
        if not base:
            base = {HMM_TO_PRETERMINAL.get(pos, "N")}
    out = set()
    for b in base:
        out |= LEXICAL_EXPANSION.get(b, {b})
    return out


class CYKParser:
    def __init__(self, grammar=GRAMMAR):
        # index rules by their right-hand side for fast lookup:  (B, C) -> {A}
        self.by_rhs: dict[tuple[str, str], set[str]] = {}
        for a, b, c in grammar:
            self.by_rhs.setdefault((b, c), set()).add(a)

    def parse(self, tagged: list[tuple[str, str]]) -> ParseResult:
        tagged = [(w, t) for w, t in tagged if w[0].isalnum()]  # drop punctuation
        n = len(tagged)
        words = [w for w, _ in tagged]
        pre = [preterminals_for(w, t) for w, t in tagged]
        result = ParseResult(words, pre)
        if n == 0:
            return result

        # table[i][j] = set of non-terminals spanning words i..j ; back[i][j][A] = (k, B, C)
        table = [[set() for _ in range(n)] for _ in range(n)]
        back = [[{} for _ in range(n)] for _ in range(n)]
        for i in range(n):
            table[i][i] = set(pre[i])

        for length in range(2, n + 1):              # span length
            for i in range(n - length + 1):          # span start
                j = i + length - 1                   # span end
                for k in range(i, j):                # split point
                    for b in table[i][k]:
                        for c in table[k + 1][j]:
                            for a in self.by_rhs.get((b, c), ()):
                                if a not in table[i][j]:
                                    table[i][j].add(a)
                                    back[i][j][a] = (k, b, c)
        result.table = table

        for s in START_SYMBOLS:
            if s in table[0][n - 1]:
                result.question_type = s
                result.tree = self._build_tree(back, words, 0, n - 1, s)
                break
        return result

    def _build_tree(self, back, words, i, j, sym) -> Tree:
        if i == j:
            return Tree(sym, [words[i]])
        k, b, c = back[i][j][sym]
        return Tree(sym, [self._build_tree(back, words, i, k, b), self._build_tree(back, words, k + 1, j, c)])


def table_as_rows(result: ParseResult) -> list[dict]:
    """Flatten the triangular CYK table for display: one row per non-empty cell."""
    rows = []
    for i, row in enumerate(result.table):
        for j, cell in enumerate(row):
            if j >= i and cell:
                rows.append({"span": f"{i}-{j}", "words": " ".join(result.tokens[i:j + 1]),
                             "non_terminals": ", ".join(sorted(cell))})
    return rows
