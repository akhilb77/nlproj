"""
question_answering.py
---------------------
The complete NyayaQuery pipeline:

  question -> normalise -> regex references -> HMM POS tags -> CYK parse
           -> TF-IDF retrieval (+ explicit reference boost) -> threshold check
           -> answer with source (original legal text, never generated text)
"""

from dataclasses import dataclass, field

from src.cyk_parser import CYKParser, ParseResult
from src.document_loader import load_documents
from src.hmm_pos_tagger import HMMPosTagger, content_words
from src.information_retrieval import Hit, TfidfRetriever
from src.regex_analyzer import LegalReference, extract_clause, find_references, mask_references
from src.text_preprocessor import STOP_WORDS, normalize, tokenize

MIN_SCORE = 0.08   # below this cosine score we say "no sufficiently relevant provision"


@dataclass
class Answer:
    question: str
    found: bool
    answer: str
    best: Hit | None = None
    clause_text: str | None = None
    related: list[Hit] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    references: list[LegalReference] = field(default_factory=list)
    pos_tags: list[tuple[str, str]] = field(default_factory=list)
    content_words: list[str] = field(default_factory=list)
    parse: ParseResult | None = None
    question_type: str = "SEARCH"
    retrieval_query: str = ""


class NyayaQuery:
    def __init__(self, records: list[dict] | None = None):
        self.records = records if records is not None else load_documents()
        self.tagger = HMMPosTagger()
        self.parser = CYKParser()
        self.retriever = TfidfRetriever(self.records)

    def ask(self, question: str, documents: list[str] | None = None, top_k: int = 5,
            min_score: float = MIN_SCORE) -> Answer:
        question = normalize(question)
        if not question:
            return Answer(question, False, "Please type a question.")

        # 1. Regex: explicit legal references
        refs = find_references(question)

        # 2. HMM: POS tags and content words
        tagged = self.tagger.tag(tokenize(question))
        keywords = content_words(tagged, STOP_WORDS)

        # 3. CYK: parse the question with references masked as a single REF token
        masked_tagged = self.tagger.tag(tokenize(mask_references(question)))
        parse = self.parser.parse(masked_tagged)
        qtype = parse.question_type or ("LOOKUP" if refs else "SEARCH")  # fallback when parsing fails

        # 4. TF-IDF retrieval: question + content words again (gives them double weight)
        query = question + " " + " ".join(keywords)
        hits = self.retriever.search(query, refs, top_k=top_k, documents=documents)
        result = Answer(question, False, "", references=refs, pos_tags=tagged, content_words=keywords,
                        parse=parse, question_type=qtype, hits=hits, retrieval_query=query)

        # 5. Explicit references that do not exist -> say so, never guess
        explicit = [r for r in refs if r.provision_type in ("Article", "Section", "Preamble")]
        if explicit and not any(self.retriever.find_reference(r, documents) for r in explicit):
            names = ", ".join(r.label for r in explicit)
            result.answer = f"{names} was not found in the selected document(s)."
            result.related = [h for h in hits if h.cosine >= min_score]
            return result

        # 6. Confidence check
        if not hits or hits[0].score < min_score:
            result.answer = "No sufficiently relevant provision was found for this question."
            return result

        best = hits[0]
        result.found, result.best = True, best
        result.related = hits[1:]
        for r in explicit:
            if r.clauses and self.retriever.matches(best.record, r):
                result.clause_text = extract_clause(best.record["text"], r.clauses)
        rec = best.record
        result.answer = f"{rec['label']} of the {rec['document_name']} - \"{rec['title']}\" - is the most relevant provision."
        return result
