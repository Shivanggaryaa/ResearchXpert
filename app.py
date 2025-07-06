import os
import streamlit as st
import fitz  # PyMuPDF

from text_utils import clean_text, chunk_text
from embeddings_utils import embed_texts, build_faiss_index
from qa_utils import answer_with_groq          # ← back to Groq
from scholar_utils import find_related_papers

st.set_page_config(page_title="Smart Research Paper Explainer", layout="wide")

# ── Header & CSS ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-title { text-align: center; color:#1f77b4; margin-bottom:.2rem; }
        .sub-title  { text-align: center; font-size:18px; margin-top:0; color:gray; }
        .stTextInput>div>div>input { border-radius:8px; padding:8px; }
    </style>
    <h1 class='main-title'>📄 ResearchXpert </h1>
    <p class='sub-title'>“Illuminate Research Documents with Conversational AI”</p>
    <hr style='margin-top:0'>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────
st.sidebar.title("📚 Navigation")
page = st.sidebar.selectbox("Go to:", ["🏠 Home – Upload & Chunk", "🔍 Query – Chat & Seek"])
st.sidebar.markdown("---")
st.sidebar.info("FAISS Index ✅" if "index" in st.session_state else "No index yet ❌")

# ── PDF text extraction utility ─────────────────────────────────────
def extract_text(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(p.get_text("text") for p in doc)

# ===================================================================
# 🏠 HOME – upload / chunk / embed / summary / related
# ===================================================================
if page.startswith("🏠"):
    st.header("📤 Upload & Process Research Paper")

    # Clear / Reset buttons
    clear_col, reset_col = st.columns(2)
    if clear_col.button("🧹 Clear Output"):
        for k in ("summary", "discover_results", "discover_query", "show_discovery"):
            st.session_state.pop(k, None)
    if reset_col.button("🔄 Full Reset"):
        st.session_state.clear()
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

    # File uploader
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        raw      = extract_text(uploaded.read())
        cleaned  = clean_text(raw)
        chunks   = chunk_text(cleaned, chunk_size=400, overlap=80)
        st.success(f"✅ Extracted {len(raw):,} characters → {len(chunks)} chunks")

        # Buttons row
        col1, col2, col3 = st.columns(3)
        build_bt = col1.button("⚙️ Build FAISS Index",  use_container_width=True)
        sum_bt   = col2.button("📝 Summarize Paper",    use_container_width=True)
        disc_bt  = col3.button("📖 Related Papers",     use_container_width=True)

        # Build index
        if build_bt:
            with st.spinner("🔧 Building FAISS index…"):
                idx = build_faiss_index(embed_texts(chunks))
            st.session_state.update({"chunks": chunks, "index": idx})
            st.success("✅ FAISS index built!")

        # Summarize via Groq
        if sum_bt:
            if "index" not in st.session_state:
                st.warning("⚠️ Build the index first.")
            elif os.getenv("GROQ_API_KEY") is None:
                st.warning("⚠️ GROQ_API_KEY not set.")
            else:
                with st.spinner("🤖 Summarizing with Groq…"):
                    summary, _ = answer_with_groq(
                        "Provide a concise summary of this paper.",
                        st.session_state["chunks"],
                        st.session_state["index"],
                        k=3,
                    )
                    st.session_state["summary"] = summary

        if "summary" in st.session_state:
            st.markdown("### 📄 Paper Summary")
            st.info(st.session_state["summary"])

        # Related‑paper discovery
        if disc_bt:
            if "summary" not in st.session_state:
                st.warning("⚠️ Generate a summary first.")
            else:
                st.session_state["show_discovery"] = True
                st.session_state.pop("discover_results", None)

        if st.session_state.get("show_discovery"):
            st.markdown("---")
            st.markdown("### 🔍 Discover Related Papers")

            def run_discovery():
                query = st.session_state.get("discover_query", "").strip()
                if not query:
                    st.warning("⚠️ Enter a topic before searching.")
                    return
                with st.spinner("🔎 Searching Semantic Scholar…"):
                    st.session_state["discover_results"] = find_related_papers(query, limit=5)
                if not st.session_state["discover_results"]:
                    st.warning("No related papers found.")

            st.text_input(
                "Enter a concise research topic:",
                key="discover_query",
                placeholder="e.g. Mobile‑Assisted Language Learning",
                on_change=run_discovery,
            )
            st.button("🔎 Search", on_click=run_discovery)

            for p in st.session_state.get("discover_results", []):
                st.markdown(
                    f"**{p['title']}**  \n"
                    f"*{p['authors']} – {p['year']}*  \n"
                    f"[🔗 View Paper]({p['url']})"
                )
    else:
        st.info("⬆️ Upload a PDF to begin.")

# ===================================================================
# 🔍 QUERY – Chat with Groq
# ===================================================================
elif page.startswith("🔍"):
    st.header("💬 Let's Converse...")

    # Guards
    if "index" not in st.session_state:
        st.warning("Build an index first in the Home tab.")
        st.stop()
    if os.getenv("GROQ_API_KEY") is None:
        st.warning("GROQ_API_KEY not set.")
        st.stop()

    chat_hist = st.session_state.setdefault("chat_history", [])

    if st.button("🧹 Clear Chat"):
        chat_hist.clear()
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

    # Render history
    for m in chat_hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # New question
    user_q = st.chat_input("Ask a question about this paper…")
    if user_q:
        chat_hist.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Groq is thinking…"):
                try:
                    answer, hits = answer_with_groq(
                        user_q,
                        st.session_state["chunks"],
                        st.session_state["index"],
                        k=3,
                    )
                except Exception as err:
                    answer, hits = f"⚠️ Error: {err}", []
            st.markdown(answer)

            if hits:
                with st.expander("🔍 Source Chunks"):
                    for rk, (idx, score) in enumerate(hits, 1):
                        st.markdown(f"**[{rk}] similarity {score:.3f}**")
                        st.code(st.session_state["chunks"][idx][:1000] + "…")

        chat_hist.append({"role": "assistant", "content": answer})
