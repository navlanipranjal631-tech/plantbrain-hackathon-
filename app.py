import streamlit as st
import google.generativeai as genai
import os
import time
import re
import numpy as np
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from groq import Groq, RateLimitError as GroqRateLimitError

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="PlantBrain — Industrial Knowledge Intelligence",
    page_icon="⛭",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FOLDER = "data"
EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "models/gemini-flash-latest"
GROQ_MODEL = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None

def call_with_retry(func, max_retries=3, base_delay=8):
    """Retries a Gemini API call with backoff if the free-tier rate limit is hit."""
    for attempt in range(max_retries):
        try:
            return func()
        except ResourceExhausted:
            if attempt < max_retries - 1:
                wait_time = base_delay * (attempt + 1)
                st.warning(f"Free-tier rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

def call_groq_with_retry(func, max_retries=3, base_delay=6):
    """Retries a Groq API call with backoff if its free-tier rate limit is hit."""
    for attempt in range(max_retries):
        try:
            return func()
        except GroqRateLimitError:
            if attempt < max_retries - 1:
                wait_time = base_delay * (attempt + 1)
                st.warning(f"Groq free-tier rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* ---- Fix: sidebar collapse/reopen arrow must always stay visible ---- */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #16233A !important;
    color: #16233A !important;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

html { color-scheme: light !important; }
.stApp { background-color: #F7F8FA !important; color-scheme: light !important; }

/* ---- Brand header ---- */
.brand-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0 18px 0;
    border-bottom: 2px solid #1F3A5F;
    margin-bottom: 4px;
}
.brand-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #16233A;
    margin: 0;
    letter-spacing: -0.5px;
}
.brand-name span { color: #C97A1A; }
.brand-tagline { color: #5B6472; font-size: 0.95rem; margin-top: 2px; }
.status-pill {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.75rem;
    color: #2E7D4F;
    border: 1px solid #BEE3CA;
    background: #EAF7EF;
    padding: 6px 14px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.status-dot {
    display: inline-block; width: 7px; height: 7px;
    background: #2E7D4F; border-radius: 50%; margin-right: 6px;
}

/* ---- Purpose box ---- */
.purpose-box {
    background: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-left: 4px solid #C97A1A;
    border-radius: 6px;
    padding: 18px 22px;
    margin: 20px 0 24px 0;
}
.purpose-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    color: #C97A1A;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 6px;
}
.purpose-text { color: #2D3543 !important; font-size: 0.96rem; line-height: 1.55; }

/* ---- Metric cards ---- */
.metric-row { display: flex; gap: 14px; margin-bottom: 24px; }
.metric-card {
    flex: 1; background: #FFFFFF; border: 1px solid #E4E7EC;
    border-radius: 8px; padding: 14px 18px;
}
.metric-label {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 0.68rem;
    color: #8A93A3; letter-spacing: 1px; text-transform: uppercase;
}
.metric-value {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 1.5rem;
    color: #16233A; font-weight: 700; margin-top: 2px;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background-color: #16233A; }
section[data-testid="stSidebar"] * { color: #D6DCE5 !important; }
section[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #E8A857 !important; font-size: 0.8rem !important;
    letter-spacing: 1px; text-transform: uppercase;
}
section[data-testid="stSidebar"] hr { border-color: #2A3B57; }
section[data-testid="stSidebar"] .stButton>button {
    background-color: #C97A1A; color: white; border: none;
    border-radius: 5px; font-weight: 600;
}

/* ---- Search bar (top, prominent) ---- */
.search-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.8rem;
    color: #16233A;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
div[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 2px solid #1F3A5F !important;
    color: #16233A !important;
    font-size: 1rem !important;
    padding: 12px 14px !important;
    border-radius: 8px !important;
}
div[data-testid="stTextInput"] input::placeholder { color: #9AA3B0 !important; }
.stForm button {
    background-color: #1F3A5F !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
}

/* ---- Selectbox (dropdown) ---- */
div[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    border: 2px solid #1F3A5F !important;
    border-radius: 8px !important;
    color: #16233A !important;
}
div[data-testid="stSelectbox"] * { color: #16233A !important; }
div[data-baseweb="popover"] { background-color: #FFFFFF !important; }
li[role="option"] {
    background-color: #FFFFFF !important;
    color: #16233A !important;
}
li[role="option"]:hover { background-color: #FFF4E4 !important; }

/* ---- Checkbox ---- */
div[data-testid="stCheckbox"] label span[data-testid="stMarkdownContainer"] p {
    color: #16233A !important;
    font-weight: 500;
}
div[data-testid="stCheckbox"] svg { color: #C97A1A !important; }

/* ---- Buttons (non-form) ---- */
.stButton>button {
    background-color: #1F3A5F !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #5B6472 !important;
    font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1F3A5F !important;
    border-bottom-color: #C97A1A !important;
}
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid #E4E7EC;
}

/* ---- About box (per-feature explanation) ---- */
.about-box {
    background: #F1F5F9;
    border: 1px solid #DCE3EC;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 4px 0 20px 0;
}
.about-box-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.05rem;
    color: #1F3A5F;
    letter-spacing: 0px;
    text-transform: none;
    font-weight: 700;
    margin-bottom: 8px;
}
.about-box-text { color: #2D3543 !important; font-size: 1rem; line-height: 1.6; }
.about-box-text * { color: #2D3543 !important; }

/* ---- Answer card ---- */
.answer-card {
    background: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-left: 4px solid #1F3A5F;
    border-radius: 8px;
    padding: 20px 22px;
    margin-top: 18px;
}
.answer-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    color: #1F3A5F;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 8px;
}
.answer-text {
    color: #16233A !important;
    font-size: 1rem;
    line-height: 1.65;
}
.answer-text * { color: #16233A !important; }
.answer-text code,
code {
    background-color: #FFF4E4 !important;
    color: #7A4A10 !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88em !important;
    border: 1px solid #F0D2A0 !important;
}
.answer-text pre,
pre {
    background-color: #FFF4E4 !important;
    border: 1px solid #F0D2A0 !important;
}

/* ---- Progress label during search ---- */
.progress-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    color: #5B6472;
    margin: 14px 0 6px 0;
    letter-spacing: 0.3px;
}
.stProgress > div > div > div {
    background-color: #C97A1A !important;
}

/* ---- Fix: st.metric widgets (Documents/Entities/Connections etc.) ---- */
[data-testid="stMetricValue"] {
    color: #16233A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #5B6472 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] * { color: #5B6472 !important; }
[data-testid="stMetricValue"] * { color: #16233A !important; }

/* ---- Checkbox: strong contrast fix ---- */
div[data-testid="stCheckbox"] { background: transparent; }
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] label span {
    color: #16233A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* ---- Left navigation ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16233A 0%, #1B2C48 100%);
    min-width: 300px !important;
}

/* Force the sidebar to always stay visible even if Streamlit marks it
   "collapsed" (it hides it via transform/margin, not width/display) */
section[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    margin-left: 0px !important;
    visibility: visible !important;
    width: 300px !important;
    min-width: 300px !important;
}
div[data-testid="stSidebarUserContent"] { visibility: visible !important; }

section[data-testid="stSidebar"] .stButton>button {
    background-color: transparent !important;
    color: #C7CFDB !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    width: 100%;
    margin-bottom: 4px;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background-color: rgba(201,122,26,0.15) !important;
    color: #F0B15E !important;
    border: 1px solid rgba(201,122,26,0.35) !important;
}
.nav-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFFFFF;
    padding: 6px 6px 2px 6px;
}
.nav-logo span { color: #E8A857; }
.nav-tagline {
    color: #8592A6;
    font-size: 0.78rem;
    padding: 0 6px 20px 6px;
    line-height: 1.4;
}
.nav-active-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.68rem;
    color: #E8A857;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 6px 10px 6px;
}

/* ---- Section header banners (one accent color per feature) ---- */
.section-banner {
    border-radius: 12px;
    padding: 26px 30px;
    margin-bottom: 22px;
    color: #FFFFFF !important;
}
.section-banner * { color: #FFFFFF !important; }
.section-banner.banner-about { background: linear-gradient(120deg, #16233A 0%, #2C4A73 100%); }
.section-banner.banner-ask { background: linear-gradient(120deg, #0F5C4F 0%, #1C8A76 100%); }
.section-banner.banner-graph { background: linear-gradient(120deg, #8A4B0F 0%, #C97A1A 100%); }
.section-banner.banner-compliance { background: linear-gradient(120deg, #7A1F1F 0%, #B0392E 100%); }
.section-banner-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.section-banner-sub { font-size: 1rem; opacity: 0.92; line-height: 1.5; }

/* ---- Human-impact / story card ---- */
.impact-card {
    background: #FFF4E4;
    border: 1px solid #F0D2A0;
    border-left: 5px solid #C97A1A;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
}
.impact-card, .impact-card * { color: #4A3410 !important; }
.impact-card b { color: #7A4A10 !important; }

.example-card {
    background: #EAF2F8;
    border: 1px solid #C3D9EA;
    border-left: 5px solid #2C5F7C;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 14px 0;
}
.example-card, .example-card * { color: #16233A !important; }

.stat-strip { display: flex; gap: 14px; margin: 18px 0; flex-wrap: wrap; }
.stat-chip {
    background: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 8px;
    padding: 12px 18px;
    flex: 1;
    min-width: 140px;
}
.stat-chip-num {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #B0392E !important;
}
.stat-chip-label { font-size: 0.78rem; color: #5B6472 !important; margin-top: 2px; }


.compliance-header {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #16233A;
    margin-top: 40px;
    margin-bottom: 2px;
}
.compliance-sub { color: #5B6472; font-size: 0.92rem; margin-bottom: 16px; }
.verdict-badge {
    display: inline-block;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0px;
    padding: 10px 20px;
    border-radius: 8px;
    margin-bottom: 14px;
}
.verdict-violation { background-color: #FDECEC !important; color: #B02A2A !important; border: 1px solid #F0B4B4; }
.verdict-atrisk { background-color: #FFF4E4 !important; color: #A9631A !important; border: 1px solid #F0D2A0; }
.verdict-compliant { background-color: #EAF7EF !important; color: #2E7D4F !important; border: 1px solid #BEE3CA; }
.compliance-card {
    background: #FFFFFF !important;
    border: 1px solid #E4E7EC;
    border-radius: 8px;
    padding: 20px 22px;
    margin-top: 10px;
    color: #16233A !important;
    font-size: 1rem;
    line-height: 1.65;
}
.compliance-card, .compliance-card p, .compliance-card div,
.compliance-card span, .compliance-card b, .compliance-card strong {
    color: #16233A !important;
    opacity: 1 !important;
}

.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.doc-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    background-color: #FFF4E4 !important;
    border: 1px solid #F0D2A0;
    color: #A9631A !important;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 0.3px;
}
.doc-tag, .doc-tag * { color: #A9631A !important; }

/* ---- Expander (View source) content ---- */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E4E7EC;
    border-radius: 6px;
}
[data-testid="stExpander"] summary {
    color: #16233A !important;
    background-color: #FFFFFF !important;
}
[data-testid="stExpander"] summary * { color: #16233A !important; }
[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    background-color: #FFFFFF !important;
}
[data-testid="stExpander"] pre {
    background-color: #F7F8FA !important;
    color: #16233A !important;
    border: 1px solid #E4E7EC;
}
[data-testid="stExpander"] pre code,
[data-testid="stExpander"] pre span {
    color: #16233A !important;
    background-color: transparent !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] div {
    color: #16233A !important;
}
[data-testid="stExpander"] svg {
    fill: #16233A !important;
}
[data-testid="stExpanderToggleIcon"] {
    color: #16233A !important;
}

/* ---- History section ---- */
.history-heading {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.75rem;
    color: #8A93A3;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 28px 0 10px 0;
}
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 6px; }

hr { border-color: #E4E7EC; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# RAG PIPELINE
# ---------------------------------------------------------------------------
CACHE_FOLDER = ".embedding_cache"
CACHE_EMBEDDINGS_FILE = os.path.join(CACHE_FOLDER, "embeddings.npy")
CACHE_FILENAMES_FILE = os.path.join(CACHE_FOLDER, "filenames.npy")
CACHE_DOCS_FILE = os.path.join(CACHE_FOLDER, "documents.npy")

@st.cache_resource(show_spinner=False)
def load_and_embed_documents():
    documents, filenames = [], []
    for filename in sorted(os.listdir(DATA_FOLDER)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read())
                filenames.append(filename)

    # If a valid cache already exists on disk (same number of documents),
    # load it instantly instead of calling the embedding API again.
    if (os.path.exists(CACHE_EMBEDDINGS_FILE)
            and os.path.exists(CACHE_FILENAMES_FILE)
            and os.path.exists(CACHE_DOCS_FILE)):
        cached_filenames = list(np.load(CACHE_FILENAMES_FILE, allow_pickle=True))
        if cached_filenames == filenames:
            cached_docs = list(np.load(CACHE_DOCS_FILE, allow_pickle=True))
            cached_embeddings = np.load(CACHE_EMBEDDINGS_FILE)
            return cached_docs, cached_filenames, cached_embeddings

    # Otherwise, compute embeddings via the API (only happens once,
    # or whenever a document is added/changed) and save them to disk.
    embeddings = []
    for doc in documents:
        result = call_with_retry(
            lambda d=doc: genai.embed_content(model=EMBED_MODEL, content=d, task_type="retrieval_document")
        )
        embeddings.append(result["embedding"])

    embeddings_array = np.array(embeddings)

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    np.save(CACHE_EMBEDDINGS_FILE, embeddings_array)
    np.save(CACHE_FILENAMES_FILE, np.array(filenames, dtype=object))
    np.save(CACHE_DOCS_FILE, np.array(documents, dtype=object))

    return documents, filenames, embeddings_array

def find_relevant_docs(query, documents, filenames, doc_embeddings, top_k=2):
    result = call_with_retry(
        lambda: genai.embed_content(model=EMBED_MODEL, content=query, task_type="retrieval_query")
    )
    query_embedding = result["embedding"]
    query_vec = np.array(query_embedding)

    similarities = []
    for i, doc_vec in enumerate(doc_embeddings):
        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        similarities.append((similarity, i))

    similarities.sort(reverse=True)
    top_results = similarities[:top_k]

    return [{"filename": filenames[idx], "content": documents[idx], "score": round(float(score), 3)}
            for score, idx in top_results]

def generate_answer(query, relevant_docs):
    context = "\n\n---\n\n".join([f"Source: {d['filename']}\n{d['content']}" for d in relevant_docs])
    prompt = f"""You are an industrial plant knowledge assistant. Answer the
question ONLY using the context documents provided below. Always mention
which source document(s) you used. If the answer cannot be found in the
context, say so clearly instead of guessing.

Formatting rules: Write source filenames as plain text, never wrapped in
backticks or code formatting. Do not use markdown code blocks anywhere
in your answer.

CONTEXT DOCUMENTS:
{context}

QUESTION: {query}

ANSWER (mention source filenames in plain text, not as code):"""

    if groq_client:
        response = call_groq_with_retry(lambda: groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        ))
        return response.choices[0].message.content
    else:
        model = genai.GenerativeModel(CHAT_MODEL)
        response = call_with_retry(lambda: model.generate_content(prompt))
        return response.text

# ---------------------------------------------------------------------------
# KNOWLEDGE GRAPH — entity extraction (no API calls, fully deterministic)
# ---------------------------------------------------------------------------
ENTITY_PATTERNS = [
    r"\bPump-\d+\b",
    r"\bCompressor-\d+\b",
    r"\bMotor-\d+\b",
    r"\bValve-\d+\b",
    r"\bWO-\d+\b",
    r"\bSP-\d+\b",
    r"\bIR-\d+\b",
    r"\bNM-\d+\b",
    r"\bIRC-\d+\b",
    r"\bGD-[A-Z]-\d+\b",
    r"\bIS\s?10816\b",
    r"\bOISD-?STD-?105\b",
    r"\bZone\s[A-C]\b",
]

@st.cache_resource(show_spinner=False)
def build_knowledge_graph(documents, filenames):
    """Extracts equipment tags, record IDs, and regulation references from
    each document via regex (no API calls needed) and builds a bipartite
    graph connecting documents to the entities they mention."""
    graph = nx.Graph()
    entity_doc_map = {}

    for filename, content in zip(filenames, documents):
        short_name = filename.replace(".txt", "")
        graph.add_node(short_name, kind="document")

        found_entities = set()
        for pattern in ENTITY_PATTERNS:
            for match in re.findall(pattern, content, flags=re.IGNORECASE):
                found_entities.add(match.strip())

        for entity in found_entities:
            if entity not in graph:
                graph.add_node(entity, kind="entity")
            graph.add_edge(short_name, entity)
            entity_doc_map.setdefault(entity, []).append(short_name)

    return graph, entity_doc_map

def render_knowledge_graph(graph, min_connections=2):
    """Renders a focused subgraph: only entities that appear in 2+ documents
    are shown (the genuinely interesting cross-references). Documents with
    no qualifying entities are dropped from the visual to reduce clutter."""
    entity_nodes = [
        n for n, a in graph.nodes(data=True)
        if a.get("kind") == "entity" and graph.degree(n) >= min_connections
    ]
    doc_nodes = set()
    for e in entity_nodes:
        doc_nodes.update(graph.neighbors(e))

    sub = graph.subgraph(list(doc_nodes) + entity_nodes).copy()

    net = Network(height="620px", width="100%", bgcolor="#FFFFFF", font_color="#16233A")
    net.barnes_hut(gravity=-6000, central_gravity=0.15, spring_length=220, spring_strength=0.015, damping=0.5)

    for node, attrs in sub.nodes(data=True):
        connections = sub.degree(node)
        if attrs.get("kind") == "document":
            net.add_node(
                node, label=node, color="#1F3A5F", shape="dot",
                size=16 + connections * 3,
                font={"size": 15, "color": "#16233A", "face": "IBM Plex Mono"},
                title=f"Document: {node}  ({connections} linked entities)"
            )
        else:
            neighbors = ", ".join(sorted(graph.neighbors(node)))
            net.add_node(
                node, label=node, color="#C97A1A", shape="dot",
                size=18 + connections * 5,
                font={"size": 17, "color": "#7A4A10", "face": "IBM Plex Mono", "bold": True},
                title=f"Entity: {node}  —  appears in: {neighbors}"
            )

    for src, dst in sub.edges():
        net.add_edge(src, dst, color="#D8DCE3", width=1.5)

    net.set_options("""
    {
      "physics": {
        "stabilization": { "iterations": 200 },
        "minVelocity": 0.75
      },
      "interaction": { "hover": true, "tooltipDelay": 100 },
      "edges": { "smooth": false }
    }
    """)

    return net, sub



def run_compliance_check(scenario, documents, filenames, doc_embeddings):
    """Runs a structured compliance check: retrieves relevant regulations,
    procedures and incident records, then asks the model for a verdict."""
    relevant_docs = find_relevant_docs(scenario, documents, filenames, doc_embeddings, top_k=4)
    context = "\n\n---\n\n".join([f"Source: {d['filename']}\n{d['content']}" for d in relevant_docs])

    prompt = f"""You are a compliance auditing agent for an industrial plant.
Given a scenario and the plant's procedures, regulations, and historical
records below, determine whether the scenario represents a compliance
VIOLATION, an AT-RISK condition (not a clear violation, but a recognized
risk pattern), or is COMPLIANT.

Formatting rules: Do not use backticks or code formatting anywhere.
Write source filenames as plain text.

Respond in EXACTLY this structure:

VERDICT: [VIOLATION / AT-RISK / COMPLIANT]

REASONING:
[2-4 sentences explaining your reasoning, citing specific source documents by name]

RECOMMENDED ACTION:
[1-2 sentences on what should be done]

SCENARIO:
{scenario}

REFERENCE DOCUMENTS:
{context}
"""
    if groq_client:
        response = call_groq_with_retry(lambda: groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        ))
        return response.choices[0].message.content, relevant_docs
    else:
        model = genai.GenerativeModel(CHAT_MODEL)
        response = call_with_retry(lambda: model.generate_content(prompt))
        return response.text, relevant_docs


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
placeholder = st.empty()
placeholder.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;color:#8A93A3;">Indexing plant knowledge base...</p>',
    unsafe_allow_html=True
)
documents, filenames, doc_embeddings = load_and_embed_documents()
placeholder.empty()

# ---------------------------------------------------------------------------
# NAVIGATION STATE
# ---------------------------------------------------------------------------
if "active_section" not in st.session_state:
    st.session_state.active_section = "about"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}
if "compliance_cache" not in st.session_state:
    st.session_state.compliance_cache = {}

NAV_ITEMS = [
    ("about", "🏭  About PlantBrain"),
    ("ask", "💬  Ask PlantBrain"),
    ("graph", "🕸️  Knowledge Graph"),
    ("compliance", "🛡️  Compliance Checker"),
]

# ---------------------------------------------------------------------------
# LEFT SIDEBAR — NAVIGATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="nav-logo">Plant<span>Brain</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nav-tagline">Unified Industrial Knowledge Intelligence<br>'
        'ET AI Hackathon 2026</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="nav-active-label">Where to go</div>', unsafe_allow_html=True)

    for key, label in NAV_ITEMS:
        prefix = "▶ " if st.session_state.active_section == key else "　 "
        if st.button(prefix + label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_section = key
            st.rerun()

    st.markdown("---")
    if st.session_state.active_section == "ask" and st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

section = st.session_state.active_section

# =============================================================================
# SECTION 1 — ABOUT PLANTBRAIN
# =============================================================================
if section == "about":
    st.markdown("""
    <div class="section-banner banner-about">
        <div class="section-banner-title">🏭 What is PlantBrain?</div>
        <div class="section-banner-sub">
        PlantBrain is an AI system that reads every scattered document inside an industrial
        plant — maintenance logs, safety procedures, inspection reports, and government
        regulations — and turns them into one place anyone can ask questions to, in seconds,
        with a source cited every time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-box-title" style="font-size:0.85rem; margin-top:6px;">Why we built this</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-box">
        <div class="about-box-text">
        In large industrial plants, critical safety information is split across 7 to 12
        disconnected systems — one place for maintenance records, another for safety
        procedures, another for inspection reports, another for regulations. Engineers spend
        up to <b>35% of their working time</b> just searching for information that already
        exists somewhere in the organisation. That delay is not just inefficient — in a plant
        handling flammable gases and heavy machinery, it can be the difference between a
        near-miss and a fatality.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-box-title" style="font-size:0.85rem;">Who this actually protects</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="impact-card">
        It is easy to talk about industrial safety in the abstract — compliance scores,
        audit checklists, quarterly reports. But the people most exposed to these risks are
        rarely the executives reading those reports. They are <b>shift workers, contract
        laborers, and maintenance technicians</b> — often on daily wages, often without the
        power to question a rushed permit or a skipped inspection. In FY2023 alone, India
        recorded over <b>6,500 fatal workplace accidents</b> (DGFASLI) — and that figure
        excludes most mining and construction deaths. In January 2025, eight workers died
        at a steel plant when a gas explosion occurred despite functioning sensors — because
        no system connected the warning signal to a decision in time. PlantBrain exists so
        that the information needed to prevent that outcome reaches the person who needs
        it, in seconds — not just the person who can afford a dashboard.
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-box-title" style="font-size:0.85rem;">A few numbers that explain why</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-strip">
        <div class="stat-chip"><div class="stat-chip-num">6,500+</div><div class="stat-chip-label">fatal workplace accidents, FY2023 (DGFASLI)</div></div>
        <div class="stat-chip"><div class="stat-chip-num">35%</div><div class="stat-chip-label">of engineer time spent just searching for information</div></div>
        <div class="stat-chip"><div class="stat-chip-num">60%</div><div class="stat-chip-label">of plants rely on manual handoffs between safety systems</div></div>
        <div class="stat-chip"><div class="stat-chip-num">7-12</div><div class="stat-chip-label">disconnected document systems per large plant</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-box-title" style="font-size:0.85rem;">What PlantBrain actually does</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="example-card">
        <b>Example 1 — Ask PlantBrain:</b> A technician asks "What is the maintenance history
        of Pump-204?" and instantly gets a timeline built from four separate work orders,
        instead of manually searching paper files or a shared drive.
    </div>
    <div class="example-card">
        <b>Example 2 — Knowledge Graph:</b> The system automatically notices that "Pump-204"
        is mentioned in four different documents spanning six months — a pattern no single
        report would reveal on its own.
    </div>
    <div class="example-card">
        <b>Example 3 — Compliance Checker:</b> Before a hot work permit is approved near a
        zone with a known gas detector outage, the system flags the conflict automatically —
        the exact combination that has preceded real industrial accidents.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="about-box-text" style="margin-top:10px; font-style:italic;">'
        'Use the menu on the left to try each feature — Ask PlantBrain, Knowledge Graph, and Compliance Checker.'
        '</div>', unsafe_allow_html=True
    )

# =============================================================================
# SECTION 2 — ASK PLANTBRAIN
# =============================================================================
elif section == "ask":
    st.markdown("""
    <div class="section-banner banner-ask">
        <div class="section-banner-title">💬 Ask PlantBrain</div>
        <div class="section-banner-sub">
        Ask any question about maintenance, safety procedures, or compliance — and get an
        answer in seconds, with the exact source document named every time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
        <div class="about-box-title">Why quick answers matter here</div>
        <div class="about-box-text">
        A delayed answer during a safety-critical moment is not a minor inconvenience — it is
        a risk multiplier. When a technician cannot quickly confirm a vibration limit, a permit
        conflict, or an equipment's repair history, decisions get made on memory and guesswork
        instead of documented fact. This feature exists to close that gap: every answer is
        grounded in an actual plant document, never invented, and the source is always shown
        so it can be independently verified.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="search-label">💬 Type your question below — no special format needed</div>', unsafe_allow_html=True)

    with st.form(key="search_form", clear_on_submit=False):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "search",
                placeholder="e.g. What is the maintenance history of Pump-204?",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Search")

    if submitted and query:
        cache_key = query.strip().lower()

        if cache_key in st.session_state.answer_cache:
            cached = st.session_state.answer_cache[cache_key]
            st.session_state.messages.insert(0, {"question": query, "answer": cached["answer"], "sources": cached["sources"]})
            st.info("⚡ Instant result — this exact question was already answered earlier in this session.")
        else:
          try:
            progress_box = st.empty()
            bar_box = st.empty()

            progress_box.markdown(
                '<div class="progress-label">🧠 Understanding your question, please hold on a moment...</div>',
                unsafe_allow_html=True
            )
            bar = bar_box.progress(15)
            time.sleep(0.4)

            progress_box.markdown(
                '<div class="progress-label">📄 Searching the plant knowledge base for relevant records...</div>',
                unsafe_allow_html=True
            )
            bar.progress(40)
            relevant_docs = find_relevant_docs(query, documents, filenames, doc_embeddings)
            bar.progress(65)

            progress_box.markdown(
                '<div class="progress-label">✍️ Drafting your answer with cited sources, almost there...</div>',
                unsafe_allow_html=True
            )
            bar.progress(85)
            answer = generate_answer(query, relevant_docs)
            bar.progress(100)

            progress_box.markdown(
                '<div class="progress-label" style="color:#2E7D4F;">✅ Done — here is what we found.</div>',
                unsafe_allow_html=True
            )
            time.sleep(0.4)
            progress_box.empty()
            bar_box.empty()

            st.session_state.answer_cache[cache_key] = {"answer": answer, "sources": relevant_docs}
            st.session_state.messages.insert(0, {"question": query, "answer": answer, "sources": relevant_docs})
          except (ResourceExhausted, GroqRateLimitError):
            progress_box.empty()
            bar_box.empty()
            st.error(
                "The free API tier allows only a limited number of requests per minute. "
                "Please wait about 30-60 seconds and try again — thank you for your patience!"
            )

    if st.session_state.messages:
        latest = st.session_state.messages[0]
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">Answer — {latest['question']}</div>
            <div class="answer-text">{latest['answer']}</div>
        </div>
        """, unsafe_allow_html=True)

        tags_html = "".join(
            [f'<span class="doc-tag">{d["filename"].split(".")[0].upper()} · {d["score"]}</span>'
             for d in latest["sources"]]
        )
        st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)

        for doc in latest["sources"]:
            with st.expander(f"View source — {doc['filename']}"):
                st.text(doc['content'])

    if len(st.session_state.messages) > 1:
        st.markdown('<div class="history-heading">Previous Questions</div>', unsafe_allow_html=True)
        for item in st.session_state.messages[1:]:
            with st.expander(item["question"]):
                st.markdown(f'<div class="answer-text">{item["answer"]}</div>', unsafe_allow_html=True)
                tags_html = "".join(
                    [f'<span class="doc-tag">{d["filename"].split(".")[0].upper()} · {d["score"]}</span>'
                     for d in item["sources"]]
                )
                st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)

    st.markdown("**Try asking:**")
    st.write("→ Maintenance history of Pump-204")
    st.write("→ Hot work permit risk near a gas detector outage")
    st.write("→ Vibration limits for cooling water pumps")
    st.write("→ Lessons from recent near-miss reports")

# =============================================================================
# SECTION 3 — KNOWLEDGE GRAPH
# =============================================================================
elif section == "graph":
    st.markdown("""
    <div class="section-banner banner-graph">
        <div class="section-banner-title">🕸️ Knowledge Graph</div>
        <div class="section-banner-sub">
        Equipment tags, work orders, and regulation references are automatically extracted
        from every document and linked into a network — revealing connections a manual
        search would never surface.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
        <div class="about-box-title">Why this feature exists</div>
        <div class="about-box-text">
        Documents mention the same equipment, permits, and regulations repeatedly — but those
        connections are invisible in a folder of separate files. This graph is built entirely
        with pattern-matching, at zero AI cost, and reveals which equipment or rules are
        referenced across multiple records — surfacing links a manual review would miss.
        </div>
    </div>
    """, unsafe_allow_html=True)

    kg_graph, kg_entity_map = build_knowledge_graph(documents, filenames)

    kcol1, kcol2, kcol3 = st.columns(3)
    kcol1.metric("Documents we read", sum(1 for _, a in kg_graph.nodes(data=True) if a.get("kind") == "document"))
    kcol2.metric("Things we found", sum(1 for _, a in kg_graph.nodes(data=True) if a.get("kind") == "entity"))
    kcol3.metric("Links between them", kg_graph.number_of_edges())

    show_graph = st.checkbox("Show interactive graph", value=False)

    if show_graph:
        net, sub = render_knowledge_graph(kg_graph, min_connections=2)
        if sub.number_of_nodes() == 0:
            st.info("No entity appears in more than one document yet — try adding more overlapping records.")
        else:
            graph_path = os.path.join(CACHE_FOLDER, "knowledge_graph.html")
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            net.save_graph(graph_path)
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_html = f.read()
            components.html(graph_html, height=640, scrolling=False)
            st.markdown(
                '<div style="color:#16233A; font-size:0.85rem; margin-top:8px; font-weight:500;">'
                '🔵 Blue = documents &nbsp;&nbsp; 🟠 Orange = equipment tags, work orders, regulations &nbsp;&nbsp; '
                '(only entities shared across 2+ documents are shown — drag nodes to explore, hover for details)'
                '</div>',
                unsafe_allow_html=True
            )

    top_shared = sorted(kg_entity_map.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    if top_shared:
        entity_html = '<div class="compliance-card" style="margin-top:14px;"><b>Most cross-referenced entities:</b><br><br>'
        for entity, docs in top_shared:
            doc_list = ", ".join(sorted(docs))
            entity_html += f'→ <b>{entity}</b> appears in {len(docs)} documents: {doc_list}<br>'
        entity_html += '</div>'
        st.markdown(entity_html, unsafe_allow_html=True)

# =============================================================================
# SECTION 4 — COMPLIANCE CHECKER
# =============================================================================
elif section == "compliance":
    st.markdown("""
    <div class="section-banner banner-compliance">
        <div class="section-banner-title">🛡️ Compliance Gap Checker</div>
        <div class="section-banner-sub">
        Select a real-world scenario. The system cross-references plant procedures,
        regulations, and historical records to automatically flag violations — before
        they reach an audit or an accident.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
        <div class="about-box-title">Why this feature exists</div>
        <div class="about-box-text">
        Most safety incidents happen when two individually-acceptable conditions overlap —
        like a hot work permit issued while a gas detector is offline. This agent cross-checks
        a scenario against procedures, regulations, and historical near-miss records to catch
        that kind of compound risk automatically, before it becomes an audit finding or an
        accident.
        </div>
    </div>
    """, unsafe_allow_html=True)

    SCENARIOS = {
        "Select a scenario...": None,
        "Hot work permit issued near a degraded gas detector (Zone A)":
            "A hot work permit was issued in Zone A within 15 meters of a gas manifold, "
            "at a time when the area's fixed gas detector was out of service and only "
            "intermittent portable readings were being taken. Is this compliant with plant "
            "procedure and regulation?",
        "Confined space entry requested while hot work is active in the same zone":
            "A confined space entry permit is requested for a vessel in a zone where a hot "
            "work permit is already active nearby, with no documented risk assessment for "
            "the combination. Is this compliant with plant procedure and regulation?",
        "Equipment condition logged as 'monitor' with no follow-up scheduled":
            "A technician logs a piece of equipment's condition as 'monitor, not yet critical' "
            "during a work order, with no automatic re-inspection scheduled before the next "
            "regular maintenance cycle three months later. Is this an acceptable practice?",
        "Pump vibration reading of 5.2 mm/s during a routine check":
            "During a routine inspection, a cooling water pump shows a vibration reading of "
            "5.2 mm/s. Is this within acceptable limits, and what action, if any, is required?",
    }

    selected_scenario_label = st.selectbox(
        "Scenario", list(SCENARIOS.keys()), label_visibility="collapsed"
    )

    run_check = st.button("Run Compliance Check")

    if run_check and SCENARIOS[selected_scenario_label]:
        verdict_text = None
        cited_docs = None
        from_cache = False

        if selected_scenario_label in st.session_state.compliance_cache:
            cached = st.session_state.compliance_cache[selected_scenario_label]
            verdict_text, cited_docs = cached["verdict_text"], cached["cited_docs"]
            from_cache = True
            st.info("⚡ Instant result — this scenario was already checked earlier in this session.")
        else:
            try:
                cprogress_box = st.empty()
                cbar_box = st.empty()

                cprogress_box.markdown(
                    '<div class="progress-label">📋 Reading the scenario and identifying relevant procedures...</div>',
                    unsafe_allow_html=True
                )
                cbar = cbar_box.progress(15)
                time.sleep(0.4)

                cprogress_box.markdown(
                    '<div class="progress-label">🔎 Cross-referencing regulations and historical records...</div>',
                    unsafe_allow_html=True
                )
                cbar.progress(45)
                verdict_text, cited_docs = run_compliance_check(
                    SCENARIOS[selected_scenario_label], documents, filenames, doc_embeddings
                )
                cbar.progress(80)

                cprogress_box.markdown(
                    '<div class="progress-label">🛡️ Finalising the compliance verdict, almost there...</div>',
                    unsafe_allow_html=True
                )
                cbar.progress(100)
                time.sleep(0.4)
                cprogress_box.empty()
                cbar_box.empty()

                st.session_state.compliance_cache[selected_scenario_label] = {
                    "verdict_text": verdict_text, "cited_docs": cited_docs
                }
            except (ResourceExhausted, GroqRateLimitError):
                cprogress_box.empty()
                cbar_box.empty()
                st.error(
                    "The free API tier allows only a limited number of requests per minute. "
                    "Please wait about 30-60 seconds and try again — thank you for your patience!"
                )
            except Exception as e:
                cprogress_box.empty()
                cbar_box.empty()
                st.error(f"Something went wrong while running the compliance check: {e}")

        if verdict_text is not None:
            verdict_text = (verdict_text or "").strip()

            if not verdict_text:
                st.error("No response was generated. Please try running the check again.")
            else:
                upper_text = verdict_text.upper()
                if "VIOLATION" in upper_text[:200]:
                    badge_class, badge_label = "verdict-violation", "⚠️ This breaks a safety rule"
                elif "AT-RISK" in upper_text[:200] or "AT RISK" in upper_text[:200]:
                    badge_class, badge_label = "verdict-atrisk", "⚡ This looks risky"
                elif "COMPLIANT" in upper_text[:200]:
                    badge_class, badge_label = "verdict-compliant", "✅ This looks safe"
                else:
                    badge_class, badge_label = "verdict-atrisk", "ℹ️ Here's what we found"

                lines = verdict_text.splitlines()
                if lines and lines[0].upper().startswith("VERDICT"):
                    body = "\n".join(lines[1:]).strip()
                else:
                    body = verdict_text

                if not body:
                    body = verdict_text

                st.markdown(f'<span class="verdict-badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="compliance-card">{body}</div>', unsafe_allow_html=True)

                tags_html = "".join(
                    [f'<span class="doc-tag">{d["filename"].split(".")[0].upper()} · {d["score"]}</span>'
                     for d in cited_docs]
                )
                st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)
    elif run_check:
        st.warning("Please select a scenario first.")