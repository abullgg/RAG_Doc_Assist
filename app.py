"""
RAG Document Assistant — Streamlit Frontend
============================================
A premium, standalone UI that communicates with the FastAPI backend
over HTTP. Does NOT import or modify any backend code.
"""

import streamlit as st
import requests
import time

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
BACKEND_URL = "http://localhost:8000"

# ------------------------------------------------------------------ #
#  Page Config & Custom CSS
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark-themed CSS injection
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --accent: #7C3AED;
    --accent-light: #A78BFA;
    --success: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --bg-card: rgba(30, 30, 46, 0.65);
    --border: rgba(124, 58, 237, 0.25);
    --text-primary: #E2E8F0;
    --text-muted: #94A3B8;
}

/* ── Global Typography ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Main container background ── */
.stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 40%, #16213E 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122A 0%, #1A1A3E 100%) !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Hero Header ── */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 50%, #C4B5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 300;
    margin-bottom: 2rem;
}

/* ── Glass Cards ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.45);
    box-shadow: 0 0 24px rgba(124, 58, 237, 0.08);
}

/* ── Answer Block ── */
.answer-block {
    background: linear-gradient(135deg, rgba(124,58,237,0.08) 0%, rgba(30,30,46,0.7) 100%);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    color: var(--text-primary);
    font-size: 1.02rem;
    line-height: 1.75;
}

/* ── Confidence Meter ── */
.confidence-container {
    text-align: center;
    padding: 1rem;
}
.confidence-value {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.confidence-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Source Chip ── */
.source-chip {
    background: rgba(30, 30, 46, 0.8);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.92rem;
    color: var(--text-primary);
    line-height: 1.6;
}
.source-chip-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-light);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* ── Status Badge ── */
.status-badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.status-online {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-offline {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
}

/* ── Text Input ── */
.stTextInput > div > div > input {
    background: rgba(30, 30, 46, 0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
}

/* ── File Uploader ── */
section[data-testid="stFileUploadDropzone"] {
    background: rgba(30, 30, 46, 0.5) !important;
    border: 2px dashed rgba(124, 58, 237, 0.3) !important;
    border-radius: 12px !important;
}

/* ── Progress Bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%) !important;
    border-radius: 999px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(124, 58, 237, 0.15) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(30, 30, 46, 0.5) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
}
.footer-text a {
    color: var(--accent-light);
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Session State Defaults
# ------------------------------------------------------------------ #
_defaults = {
    "uploaded_filename": None,
    "job_id": None,
    "last_answer": None,
    "upload_complete": False,
    "doc_count": 0,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ------------------------------------------------------------------ #
#  Helper: Backend Health Check
# ------------------------------------------------------------------ #
def check_backend() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


backend_alive = check_backend()


# ------------------------------------------------------------------ #
#  Hero Header
# ------------------------------------------------------------------ #
st.markdown('<p class="hero-title">🧠 RAG Document Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Upload documents · Ask questions · Get AI-powered answers grounded in your data</p>',
    unsafe_allow_html=True,
)

# Backend status indicator (top-right feel via columns)
col_spacer, col_status = st.columns([5, 1])
with col_status:
    if backend_alive:
        st.markdown(
            '<span class="status-badge status-online">● ONLINE</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-offline">● OFFLINE</span>',
            unsafe_allow_html=True,
        )

if not backend_alive:
    st.error(
        "**Backend is not running.** Start it with:\n\n"
        "```\npython -m uvicorn src.main:app --reload\n```"
    )
    st.stop()


# ------------------------------------------------------------------ #
#  Sidebar — Document Upload
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("### 📁 Document Upload")
    st.caption("Drag & drop or browse for PDF / TXT files")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        help="Accepted formats: .pdf, .txt",
    )

    if uploaded_file:
        st.markdown(
            f'<div class="glass-card" style="padding:1rem">'
            f'<strong style="color:#A78BFA">📄 {uploaded_file.name}</strong><br>'
            f'<span style="color:#94A3B8;font-size:0.85rem">'
            f'{uploaded_file.size / 1024:.1f} KB · {uploaded_file.type}</span></div>',
            unsafe_allow_html=True,
        )

        if st.button("🚀 Process Document", use_container_width=True):
            # Post to backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=30)

                if resp.status_code == 200:
                    data = resp.json()
                    job_id = data.get("job_id") or data.get("document_id")
                    status = data.get("status", "")
                    st.session_state.job_id = job_id
                    st.session_state.uploaded_filename = uploaded_file.name

                    if status == "completed" or status == "success":
                        # Small file — processed synchronously
                        st.session_state.upload_complete = True
                        st.session_state.doc_count += 1
                        st.success(f"✅ **{uploaded_file.name}** processed instantly!")
                    else:
                        # Large file — poll for progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        while True:
                            try:
                                poll = requests.get(
                                    f"{BACKEND_URL}/status/{job_id}", timeout=5
                                )
                                poll_data = poll.json()
                                pstatus = poll_data.get("status", "unknown")
                                progress = poll_data.get("progress", 0)

                                progress_bar.progress(min(progress / 100, 1.0))
                                status_text.markdown(
                                    f"**{pstatus.upper()}** — {progress}%"
                                )

                                if pstatus == "completed":
                                    st.session_state.upload_complete = True
                                    st.session_state.doc_count += 1
                                    st.success("✅ Document processed!")
                                    break
                                elif pstatus == "failed":
                                    st.error(
                                        f"❌ Processing failed: {poll_data.get('error')}"
                                    )
                                    break

                                time.sleep(2)
                            except Exception as poll_err:
                                st.warning(f"Polling error: {poll_err}")
                                break
                else:
                    st.error(f"Upload failed ({resp.status_code}): {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection lost to backend.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.markdown("---")

    # Stats panel
    st.markdown("### 📊 Session Stats")
    stat1, stat2 = st.columns(2)
    with stat1:
        st.metric("Documents", st.session_state.doc_count)
    with stat2:
        st.metric(
            "Status",
            "Ready" if st.session_state.upload_complete else "Idle",
        )

    if st.session_state.uploaded_filename:
        st.caption(f"Last file: **{st.session_state.uploaded_filename}**")


# ------------------------------------------------------------------ #
#  Main Area — Question & Answer
# ------------------------------------------------------------------ #
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 💬 Ask a Question")

if st.session_state.uploaded_filename:
    st.caption(f"Querying against: **{st.session_state.uploaded_filename}**")
else:
    st.info("⬅️ Upload a document using the sidebar to get started.")

question = st.text_input(
    "Your question:",
    placeholder="e.g.  What are the key topics discussed in this document?",
    disabled=(not st.session_state.upload_complete),
    label_visibility="collapsed",
)

ask_disabled = not st.session_state.upload_complete or not question

if st.button("🔍 Get Answer", disabled=ask_disabled, use_container_width=True):
    with st.spinner("Searching documents and generating answer..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question, "top_k": 3},
                timeout=120,
            )

            if resp.status_code == 200:
                data = resp.json()
                st.session_state.last_answer = data
            else:
                st.error(f"Error ({resp.status_code}): {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Lost connection to backend.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Display Answer (persists across reruns via session_state)
# ------------------------------------------------------------------ #
if st.session_state.last_answer:
    data = st.session_state.last_answer

    st.markdown("---")

    col_answer, col_conf = st.columns([4, 1])

    with col_answer:
        st.markdown("### 🎯 Answer")
        st.markdown(
            f'<div class="answer-block">{data["answer"]}</div>',
            unsafe_allow_html=True,
        )

    with col_conf:
        confidence = data.get("confidence", 0)
        pct = f"{confidence:.0%}"
        st.markdown(
            f'<div class="confidence-container">'
            f'<div class="confidence-value">{pct}</div>'
            f'<div class="confidence-label">Confidence</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Sources
    sources = data.get("sources", [])
    if sources:
        st.markdown("### 📚 Retrieved Sources")
        for i, source in enumerate(sources, 1):
            truncated = source[:600] + "…" if len(source) > 600 else source
            st.markdown(
                f'<div class="source-chip">'
                f'<div class="source-chip-header">Source {i}</div>'
                f"{truncated}</div>",
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------ #
#  Footer
# ------------------------------------------------------------------ #
st.markdown("---")
st.markdown(
    '<p class="footer-text">'
    "Built with Streamlit · FastAPI · Ollama · FAISS · Hybrid BM25+Semantic Search"
    "</p>",
    unsafe_allow_html=True,
)
