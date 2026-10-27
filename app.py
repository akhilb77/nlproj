"""
NyayaQuery - Streamlit front end.
Run:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.cyk_parser import table_as_rows
from src.document_loader import document_names
from src.question_answering import MIN_SCORE, NyayaQuery

st.set_page_config(page_title="NyayaQuery", page_icon="⚖️", layout="wide")

EXAMPLES = [
    "What is Article 21?",
    "Which Article deals with equality before law?",
    "Explain Article 19(1)(a).",
    "Which Article protects freedom of speech?",
    "What are the fundamental rights?",
    "What does Article 32 provide?",
    "Which Article abolishes untouchability?",
]


@st.cache_resource(show_spinner="Training HMM tagger and building TF-IDF index...")
def get_engine() -> NyayaQuery:
    return NyayaQuery()


engine = get_engine()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    docs = document_names(engine.records)
    choice = st.selectbox("Search in", ["All available documents"] + docs,
                          index=1 + docs.index("Constitution of India") if "Constitution of India" in docs else 0)
    selected = None if choice == "All available documents" else [choice]
    top_k = st.slider("Results to retrieve (top-k)", 3, 10, 5)
    min_score = st.slider("Similarity threshold", 0.0, 0.5, MIN_SCORE, 0.01)
    st.caption(f"{len(engine.records)} provisions loaded from {len(docs)} document(s).")
    st.markdown("---")
    st.markdown("**Algorithms**\n\n1. Regular Expressions\n2. HMM POS tagging\n3. CYK parsing\n4. TF-IDF + cosine similarity")

# ---------------- header ----------------
st.title("⚖️ NyayaQuery")
st.write("An NLP-based question answering system for the **Constitution of India** and Bare Acts. "
         "Ask a question in plain English; NyayaQuery returns the original text of the most relevant provision.")

if "question" not in st.session_state:
    st.session_state.question = ""

st.write("**Try an example:**")
cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 4].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.question = ex

with st.form("ask"):
    question = st.text_input("Your question", key="question", placeholder="e.g. Which Article protects freedom of speech?")
    submitted = st.form_submit_button("Ask", type="primary")

if not st.session_state.question:
    st.stop()

ans = engine.ask(st.session_state.question, documents=selected, top_k=top_k, min_score=min_score)

# ---------------- answer ----------------
st.subheader("Answer")
if ans.found:
    rec = ans.best.record
    st.success(ans.answer)
    c1, c2, c3 = st.columns(3)
    c1.metric("Provision", rec["label"])
    c2.metric("Similarity score", f"{ans.best.score:.3f}")
    c3.metric("Question type", ans.question_type)
    st.markdown(f"### {rec['label']}: {rec['title']}")
    st.caption(f"Source: {rec['document_name']}" + (f" · {rec['part']}" if rec["part"] else "")
               + (f" · {rec['heading']}" if rec["heading"] else ""))
    if ans.clause_text:
        st.info(f"**Requested clause:** {ans.clause_text}")
    st.markdown("**Original legal text**")
    st.write(rec["text"])
else:
    st.warning(ans.answer)

if ans.related:
    st.subheader("Related provisions")
    for h in ans.related:
        r = h.record
        with st.expander(f"{r['label']} — {r['title']}  (score {h.score:.3f})"):
            st.caption(r["document_name"] + (f" · {r['part']}" if r["part"] else ""))
            st.write(r["text"])

# ---------------- NLP analysis ----------------
with st.expander("🔍 NLP analysis (how the answer was found)", expanded=False):
    st.markdown("#### 1. Regular expressions — detected legal references")
    if ans.references:
        st.table(pd.DataFrame([{"reference": r.label, "type": r.provision_type, "number": r.number,
                                "clauses": "".join(f"({c})" for c in r.clauses), "matched text": r.raw}
                               for r in ans.references]))
    else:
        st.write("No explicit Article / Section reference found.")

    st.markdown("#### 2. HMM POS tags")
    st.dataframe(pd.DataFrame([{"word": w, "tag": t, "known word": engine.tagger.is_known(w)} for w, t in ans.pos_tags]),
                 hide_index=True, use_container_width=True)
    st.write("Content words used to strengthen retrieval:", ", ".join(ans.content_words) or "—")

    st.markdown("#### 3. CYK parse")
    p = ans.parse
    st.write("Parser input:", " ".join(p.tokens) if p.tokens else "—")
    if p.success:
        st.write(f"Parsed successfully. Question classified as **{p.question_type}**.")
        st.code(p.tree.pformat(margin=100), language="text")
    else:
        st.write(f"Grammar could not parse this question → fallback classification **{ans.question_type}**; "
                 "retrieval still runs normally.")
    if p.table:
        st.dataframe(pd.DataFrame(table_as_rows(p)), hide_index=True, use_container_width=True)

    st.markdown("#### 4. TF-IDF retrieval")
    st.write("Query sent to the retriever:", f"`{ans.retrieval_query}`")
    st.dataframe(pd.DataFrame([{"provision": h.record["label"], "title": h.record["title"],
                                "cosine": round(h.cosine, 4), "boost": h.boost, "final score": round(h.score, 4)}
                               for h in ans.hits]), hide_index=True, use_container_width=True)
