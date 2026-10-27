"""
regex_analyzer.py
-----------------
Algorithm 1: Regular Expressions (a regex is a compact description of a
Finite State Automaton).  Detects explicit legal references such as

    Article 21, Art. 19(1)(a), article 14, Articles 14 and 21,
    Section 302, Sec. 420, s. 34, Part III, Preamble
"""

import re
from dataclasses import dataclass, field

# keyword -> canonical provision type
KEYWORDS = {
    "article": "Article", "articles": "Article", "art": "Article", "arts": "Article",
    "section": "Section", "sections": "Section", "sec": "Section", "s": "Section", "u/s": "Section",
}

NUMBER = r"\d{1,3}[A-Z]{0,2}"              # 21, 21A, 243ZH
CLAUSES = r"(?:\s*\(\s*\w{1,4}\s*\))*"     # (1)(a), ( 1 ) ( a )

# keyword, then one number, then optional ", 15 and 16" style continuations
REF_RE = re.compile(
    rf"\b(?P<kw>articles?|arts?|sections?|secs?|u/s|s)\.?\s*"
    rf"(?P<nums>{NUMBER}{CLAUSES}(?:\s*(?:,|and|or|&|to)\s*{NUMBER}{CLAUSES})*)",
    re.IGNORECASE,
)
ONE_NUM_RE = re.compile(rf"(?P<num>{NUMBER})(?P<cl>{CLAUSES})", re.IGNORECASE)
PART_RE = re.compile(r"\bpart\s+(?P<num>[IVXL]+[A-C]?|\d{1,2}[A-C]?)\b", re.IGNORECASE)
PREAMBLE_RE = re.compile(r"\bpreamble\b", re.IGNORECASE)


@dataclass
class LegalReference:
    provision_type: str           # "Article", "Section", "Part", "Preamble"
    number: str                   # "19", "302", "III", "Preamble"
    clauses: list[str] = field(default_factory=list)   # ["1", "a"]
    raw: str = ""
    span: tuple[int, int] = (0, 0)

    @property
    def label(self) -> str:
        cl = "".join(f"({c})" for c in self.clauses)
        return self.number if self.provision_type == "Preamble" else f"{self.provision_type} {self.number}{cl}"


def find_references(text: str) -> list[LegalReference]:
    refs = []
    for m in REF_RE.finditer(text):
        kw = m.group("kw").lower()
        # a bare "s" must be written "s." or "s 302" to count; avoid matching words like "rights 5"
        if kw == "s" and not re.match(r"s\.?\s*\d", m.group(0), re.I):
            continue
        ptype = KEYWORDS[kw]
        for n in ONE_NUM_RE.finditer(m.group("nums")):
            clauses = re.findall(r"\(\s*(\w{1,4})\s*\)", n.group("cl"))
            refs.append(LegalReference(ptype, n.group("num").upper(), clauses, m.group(0).strip(), m.span()))
    for m in PART_RE.finditer(text):
        refs.append(LegalReference("Part", m.group("num").upper(), [], m.group(0), m.span()))
    for m in PREAMBLE_RE.finditer(text):
        refs.append(LegalReference("Preamble", "Preamble", [], m.group(0), m.span()))
    return refs


def mask_references(text: str, placeholder: str = "REF") -> str:
    """Replace each Article/Section reference by a single token (used by the CYK parser)."""
    return REF_RE.sub(lambda m: f" {placeholder} ", text)


def extract_clause(text: str, clauses: list[str]) -> str | None:
    """Return the text of clause (1)(a) etc. from a provision, or None if not found."""
    part = text
    for c in clauses:
        level = r"\d+[A-Z]?" if c[0].isdigit() else r"[a-z]{1,4}"
        start = re.search(rf"\({re.escape(c)}\)\s", part)
        if not start:
            return None
        rest = part[start.start():]
        nxt = re.search(rf"\s\((?!{re.escape(c)}\))({level})\)\s", rest[len(c) + 2:])
        part = rest[: nxt.start() + len(c) + 2] if nxt else rest
    return part.strip()
