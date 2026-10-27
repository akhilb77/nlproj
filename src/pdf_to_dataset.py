"""
pdf_to_dataset.py
-----------------
One-time utility: converts the official PDF of the Constitution of India
into a structured dataset (JSON) that the rest of NyayaQuery uses.

Why is this tricky?  The PDF contains, besides the legal text:
  * running page headers  ("THE CONSTITUTION OF INDIA (Part III...)")
  * footnotes at the bottom of every page (amendment history)
  * amendment markers inside the text  ("4[the sovereignty ...]")
  * omitted/repealed Articles

Strategy
  1. Extract text page by page with `pdftotext -layout` (poppler).
  2. Parse the PDF's own TABLE OF CONTENTS -> authoritative list of
     Article numbers, titles, Parts and group headings.
  3. Clean the body pages (remove headers, footnotes, markers).
  4. For every Article in the contents, locate its start in the body text
     and cut the text up to the next Article.
  5. Validate: every Article from the contents must be found in the body.

Usage:
    python -m src.pdf_to_dataset path/to/constitution.pdf
"""

import json
import re
import subprocess
import sys
from pathlib import Path

DOC_NAME = "Constitution of India"


# ----------------------------------------------------------------------
# Step 1: PDF -> list of page strings
# ----------------------------------------------------------------------
def extract_pages(pdf_path: str) -> list[str]:
    """Run pdftotext (keeps layout) and split into pages on form-feed."""
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.split("\f")


def find_body_range(pages: list[str]) -> tuple[int, int, int]:
    """Return (contents_start, body_start, body_end) page indexes.

    body_start = page containing the Preamble
    body_end   = page containing Article 395 (the last Article)
    """
    body_start = next(i for i, p in enumerate(pages) if "WE, THE PEOPLE OF INDIA" in p and i > 5)
    body_end = next(i for i, p in enumerate(pages)
                    if re.search(r"^\s*395\.\s+Repeals", p, re.M) and i > body_start)
    contents_start = next(i for i, p in enumerate(pages) if "CONTENTS" in p)
    return contents_start, body_start, body_end


# ----------------------------------------------------------------------
# Step 2: parse table of contents
# ----------------------------------------------------------------------
ENTRY_RE = re.compile(r"^\s*(\[)?\s*(\d{1,3}[A-Z]{0,2})\.\s+(.*)$")
PART_RE = re.compile(r"^\s*PART\s+([IVX]+A?B?)\s*$")


def parse_contents(content_pages: list[str]) -> list[dict]:
    """Return list of {number, title, part, heading} from the contents pages."""
    entries = []
    part_no, part_title_lines = "", []
    heading_lines: list[str] = []
    collecting_part_title = False
    current = None  # entry whose title is still being continued

    for page in content_pages:
        for raw in page.splitlines():
            line = raw.strip()
            if not line:
                continue
            # page furniture such as "(iii)   Contents"
            if re.match(r"^\(?[ivxl]+\)?\s+Contents", line) or re.match(r"^Contents\s+\(?[ivxl]+\)?$", line):
                continue
            if line in ("ARTICLES", "ARTICLE", "CONTENTS", "THE CONSTITUTION OF INDIA", "PREAMBLE"):
                collecting_part_title = False
                continue
            if set(line) <= {"_"}:
                continue

            m_part = PART_RE.match(raw)
            if m_part:
                part_no, part_title_lines = m_part.group(1), []
                heading_lines = []
                collecting_part_title = True
                current = None
                continue

            m = ENTRY_RE.match(raw)
            if m and not (current and not current["_done"]):
                collecting_part_title = False
                bracket, number, rest = m.groups()
                current = {
                    "number": number, "title": rest.strip(),
                    "part": f"Part {part_no} - {' '.join(part_title_lines).title()}" if part_no else "",
                    "heading": " | ".join(heading_lines),
                    "omitted": bool(bracket), "_done": rest.rstrip().endswith((".", ".]")),
                }
                entries.append(current)
                continue

            # continuation of a multi-line title
            if current and not current["_done"]:
                current["title"] += " " + line
                current["_done"] = line.endswith((".", ".]"))
                continue

            # otherwise it is a Part title line or a group heading
            if collecting_part_title and line.isupper():
                part_title_lines.append(line)
            else:
                collecting_part_title = False
                # a new heading line; start a fresh heading after an article entry
                if current is not None and current["_done"]:
                    heading_lines = []
                    current = None
                heading_lines.append(line)

    for e in entries:
        e.pop("_done", None)
        e["title"] = re.sub(r"\s+", " ", e["title"].replace("\u2019", "'")).strip().rstrip(".]").rstrip()
        # Article 2A etc. are listed as "[2A. Sikkim ...—Omitted.]"
        if "Omitted" in e["title"] or "Repealed" in e["title"] or e["omitted"]:
            e["omitted"] = True
    return entries


# ----------------------------------------------------------------------
# Step 3: clean body pages
# ----------------------------------------------------------------------
HEADER_RE = re.compile(r"^\s*\d*\s*THE CONSTITUTION OF INDIA\s*$")
SUBHEADER_RE = re.compile(r"^\s*\((Part|Chapter|Art\.|Arts\.|Sch\.|Schedule)[^\n]*$")


def clean_page(page: str) -> str:
    """Remove running header, footnote block and amendment markers."""
    out = []
    for line in page.splitlines():
        # footnote block starts at the long underscore line -> drop the rest
        if re.match(r"^\s*_{10,}\s*$", line):
            break
        out.append(line)
    # drop running headers (first 4 lines of the page only)
    cleaned = []
    for i, line in enumerate(out):
        if i < 4 and (HEADER_RE.match(line) or SUBHEADER_RE.match(line)):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)

    # normalise typographic apostrophes so titles match between contents and body
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    # a footnote number sitting alone on a line (e.g. " 3" before "*239A.") -> remove the line
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$\n", "", text)
    # "*239A." : asterisk used by the printers before some Article numbers
    text = re.sub(r"(?m)^(\s*)\*(?=\d)", r"\1", text)
    # placeholder lines of omitted clauses, e.g. "(4)* * * *"
    text = re.sub(r"(?m)^\s*\(\w+\)\s*\*[\s*]*$\n?", "", text)
    # marker glued to bracket:  4[the sovereignty ...  -> [the sovereignty ...
    text = re.sub(r"(?<![\w(\-])\d{1,3}(?=\[)", "", text)
    return text


def join_body(pages: list[str]) -> str:
    """Clean pages, join them, and tidy whitespace / brackets."""
    text = "\n".join(clean_page(p) for p in pages)
    text = text.replace("[", "").replace("]", "")
    # "*[239A." -> after bracket removal the asterisk is glued to the number
    text = re.sub(r"(?m)^(\s*)\*(?=\d)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


# ----------------------------------------------------------------------
# Step 4: cut body into Articles using the contents list
# ----------------------------------------------------------------------
def title_regex(title: str) -> str:
    """Regex matching the start of the title, tolerant to spacing/line breaks."""
    words = re.findall(r"\S+", title)[:3]
    return r"\s+".join(re.escape(w) for w in words)


OMITTED_NOTE_RE = re.compile(
    r"\s\d{1,3}[A-Z]{0,2}\.\s+[A-Z][^\n]{0,300}?\.\.?\s*[\u2014-]\s*(?:Omitted|Rep\.)", re.S)


# note of a substituted/replaced article, e.g. "232. Interpretation.—Articles 230, 231 ... subs. by ..."
SUBS_NOTE_RE = re.compile(r"\s\d{1,3}[A-Z]{0,2}\.\s+[A-Z][^\n.]{0,100}\.\s*\u2014[^.]{0,120}?\bsubs\. by", re.S)


def heading_suffixes(entries: list[dict]) -> list[str]:
    """All possible tails of the group headings listed in the contents.
    Used to strip a heading that is printed at the end of the previous Article."""
    tails = set()
    for e in entries:
        lines = [l.strip() for l in e["heading"].split("|") if l.strip()]
        for i in range(len(lines)):
            tails.add(re.sub(r"\s+", " ", " ".join(lines[i:])).lower())
        # Part titles ("THE STATES") are also printed before the next Article
        if " - " in e["part"]:
            tails.add(e["part"].split(" - ", 1)[1].lower())
    return sorted(tails, key=len, reverse=True)


def trim_article_text(text: str, tails: list[str]) -> str:
    """Remove notes about omitted Articles and trailing group headings."""
    padded = " " + text
    hits = [m.start() for m in (OMITTED_NOTE_RE.search(padded), SUBS_NOTE_RE.search(padded)) if m]
    if hits:
        text = padded[:min(hits)].strip()
    # "* * *" of a repealed *Article* at the end of the text (not a clause such as "(3) (omitted)")
    text = re.sub(r"\*(?:\s*\*)+", "(omitted)", text)
    text = re.sub(r"(?<!\))\s*\d{0,2}\s*\(omitted\)\s*$", "", text).strip()
    text = re.sub(r"\s+\d{1,2}$", "", text)
    # footnote digit glued to a mid-sentence omission:  "3(omitted)" -> "(omitted)"
    text = re.sub(r"\s*\d{1,2}(?=\(omitted\))", " ", text)
    # leftover ALL-CAPS heading words after the final full stop (e.g. "... State. THE")
    text = re.sub(r"(?<=[.;:])\s+(?:[A-Z][A-Z,'\-]*\s*){1,8}$", "", text).strip()
    changed = True
    while changed:
        changed = False
        low = text.lower().rstrip()
        for t in tails:
            if len(t) > 3 and low.endswith(t) and len(text) > len(t) + 20:
                text = text[: len(low) - len(t)].rstrip()
                changed = True
                break
    # "(f)* * * *" in the PDF marks a clause that was repealed
    text = re.sub(r"\*(?:\s*\*)+", "(omitted)", text)
    return text.strip()


def split_articles(body: str, entries: list[dict]) -> tuple[list[dict], list[str]]:
    positions, missing = [], []
    cursor = 0
    for e in entries:
        if e["omitted"]:
            continue
        pat = re.compile(r"(?m)^\s*" + re.escape(e["number"]) + r"\.\s+" + title_regex(e["title"]), re.I)
        m = pat.search(body, cursor)
        if not m:
            missing.append(e["number"])
            continue
        positions.append((m.start(), m.end(), e))
        cursor = m.end()

    tails = heading_suffixes(entries)
    articles = []
    for idx, (start, _end, e) in enumerate(positions):
        stop = positions[idx + 1][0] if idx + 1 < len(positions) else len(body)
        chunk = body[start:stop]
        # remove "21. " prefix, then cut the "Title.—" part: keep only the legal text
        chunk = re.sub(r"^\s*" + re.escape(e["number"]) + r"\.\s+", "", chunk, count=1)
        text = chunk.split(".—", 1)[1] if ".—" in chunk[:400] else chunk
        # strip Part / Chapter banner lines that precede the next Article
        text = re.sub(r"(?m)^\s*(PART\s+[IVX]+A?B?|CHAPTER\s+[IVX]+.*)\s*$", "", text)
        text = re.sub(r"\s*\n\s*", " ", text).strip()
        text = trim_article_text(text, tails)
        articles.append({
            "document_name": DOC_NAME,
            "provision_type": "Article",
            "provision_number": e["number"],
            "title": e["title"],
            "part": e["part"],
            "heading": e["heading"],
            "text": text,
        })
    return articles, missing


def extract_preamble(body: str) -> dict:
    m = re.search(r"WE, THE PEOPLE OF INDIA.*?(?=\n\s*PART I\b)", body, re.S)
    text = re.sub(r"\s+", " ", m.group(0)).strip() if m else ""
    return {
        "document_name": DOC_NAME, "provision_type": "Preamble",
        "provision_number": "Preamble", "title": "Preamble to the Constitution",
        "part": "", "heading": "", "text": text,
    }


# ----------------------------------------------------------------------
def build_dataset(pdf_path: str, out_dir: str = "data") -> list[dict]:
    pages = extract_pages(pdf_path)
    c_start, b_start, b_end = find_body_range(pages)
    entries = parse_contents(pages[c_start:b_start])
    body = join_body(pages[b_start:b_end + 1])
    articles, missing = split_articles(body, entries)
    provisions = [extract_preamble(body)] + articles

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "constitution_of_india.json").write_text(
        json.dumps(provisions, indent=2, ensure_ascii=False), encoding="utf-8")

    # human-readable text version of the same data
    with open(out / "constitution_of_india.txt", "w", encoding="utf-8") as f:
        for p in provisions:
            label = "Preamble" if p["provision_type"] == "Preamble" else f"Article {p['provision_number']}"
            f.write(f"{label}. {p['title']}\n")
            if p["part"]:
                f.write(f"[{p['part']}" + (f" | {p['heading']}" if p["heading"] else "") + "]\n")
            f.write(p["text"] + "\n\n")

    print(f"Contents entries      : {len(entries)} ({sum(e['omitted'] for e in entries)} omitted/repealed)")
    print(f"Provisions extracted  : {len(provisions)} (incl. Preamble)")
    print(f"Not found in body     : {missing if missing else 'none'}")
    return provisions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pdf_to_dataset <constitution.pdf>")
        sys.exit(1)
    build_dataset(sys.argv[1])
