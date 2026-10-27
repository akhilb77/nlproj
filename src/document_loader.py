"""
document_loader.py
------------------
Loads legal documents (Constitution + any Bare Acts) from JSON files.

Every JSON file is a list of records with this schema:
    document_name, provision_type, provision_number, title, part, heading, text

To add a Bare Act, drop a JSON file with the same schema into data/bare_acts/.
No code change is needed.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REQUIRED = ("document_name", "provision_type", "provision_number", "title", "text")


def load_json_file(path: Path) -> list[dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    clean = []
    for r in records:
        if not all(k in r for k in REQUIRED):
            continue  # skip malformed records instead of crashing
        r.setdefault("part", "")
        r.setdefault("heading", "")
        r["provision_number"] = str(r["provision_number"])
        clean.append(r)
    return clean


def label(record: dict) -> str:
    """Human-readable label, e.g. 'Article 21' or 'Preamble'."""
    if record["provision_type"] == record["provision_number"]:
        return record["provision_type"]
    return f"{record['provision_type']} {record['provision_number']}"


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load the Constitution plus every JSON in data/bare_acts/."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.json")) + sorted((data_dir / "bare_acts").glob("*.json"))
    records = []
    for f in files:
        records.extend(load_json_file(f))
    for i, r in enumerate(records):
        r["id"] = i
        r["label"] = label(r)
    return records


def document_names(records: list[dict]) -> list[str]:
    return sorted({r["document_name"] for r in records})
