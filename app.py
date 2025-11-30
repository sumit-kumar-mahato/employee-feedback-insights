import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import google.generativeai as genai  # official Google SDK

# ==========================================================
# 🔐 Gemini API Key (Streamlit Secrets)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
# ==========================================================

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Employee Feedback Insights — Sentiment Summary (RAG)",
    page_icon="📘",
    layout="wide"
)

# ------------------ UI COLORS ------------------
WHITE = "#FFFFFF"
CARD_BG = "rgba(255,255,255,0.12)"
ANSWER_BG = "rgba(255,255,255,0.20)"
ACCENT = "#00D16C"
BORDER = "#00AEEF"

# ------------------ HEADER ------------------
header_html = f"""
<div style="text-align:center; padding: 25px;">
    <h1 style="color:{ACCENT}; font-size:42px; margin-bottom:5px;">
        📘 Employee Feedback Insights
    </h1>
    <h3 style="color:{WHITE}; margin-top:0px; font-weight:400;">
        RAG-powered Sentiment & Insights Extraction
    </h3>
    <p style="color:{WHITE}; font-size:16px; margin-top:8px;">
        Upload your employee feedback CSV and get a concise AI-generated sentiment summary.
    </p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ------------------ LOAD EMBEDDING MODEL ------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# ------------------ TEXT CHUNKING ------------------
def chunk_text(text, chunk_size=250, overlap=40):
    words = text.split()
    if not words:
        return []
    chunks = []
    pos = 0
    while pos < len(words):
        chunk = " ".join(words[pos:pos + chunk_size])
        chunks.append(chunk)
        pos += chunk_size - overlap
    return chunks

# ------------------ FAISS INDEX ------------------
def build_index(texts: pd.Series):
    vectors = model.encode(texts.tolist(), convert_to_numpy=True, show_progress_bar=False)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def retrieve(df_local: pd.DataFrame, index, query: str, k: int):
    k = min(k, index.ntotal)
    q_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, k)
    results = df_local.iloc[I[0]].copy()
    results["distance"] = D[0]
    return results

# ------------------ PROMPT FOR SENTIMENT SUMMARY ------------------
def build_prompt(query: str, feedback_texts):

    examples = "\n".join([f"- {txt}" for txt in feedback_texts])

    return f"""
You are an HR insights expert.

Using ONLY the retrieved feedback below, write a SHORT sentiment summary
(about 4–6 lines, no bullet points, no long essay).

QUESTION:
{query}

RETRIEVED FEEDBACK:
{examples}

Return ONLY valid JSON:

{{
  "overall_sentiment": "Positive / Negative / Mixed",
  "summary": "Short paragraph summarizing the recurring employee concerns."
}}
"""

# ------------------ GEMINI CALL (google-generativeai) ------------------
def call_gemini(prompt: str) -> dict:
    # FIXED — USE A MODEL YOU ACTUALLY HAVE
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    response = model.generate_content(prompt)
    raw = response.text.strip()

    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except:
        return {"error": "Gemini returned invalid JSON", "raw": raw}

# ==========================================================
#                     STREAMLIT APP
# ==========================================================
st.markdown(f"<h3 style='color:{ACCENT};'>📂 Step 1: Upload CSV</h3>", unsafe_allow_html=True)
st.info("CSV must contain a **feedback_text** column.")

uploaded = st.file_uploader("", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    if "feedback_text" not in df.columns:
        st.error("❌ 'feedback_text' column is missing.")
        st.stop()

    st.success(f"✔ Loaded {len(df)} feedback rows.")
    st.dataframe(df.head())

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------- CHUNKING ----------
    st.markdown(f"<h3 style='color:{ACCENT};'>🧩 Step 2: Chunking Feedback</h3>", unsafe_allow_html=True)

    rows = []
    for text in df["feedback_text"]:
        for ch in chunk_text(str(text)):
            rows.append({"chunk": ch})

    df_chunks = pd.DataFrame(rows)
    st.success(f"📌 Created **{len(df_chunks)} chunks** from {len(df)} entries.")

    # ---------- INDEX ----------
    with st.spinner("🔧 Building embedding index..."):
        index = build_index(df_chunks["chunk"])

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------- QUESTION ----------
    st.markdown(f"<h3 style='color:{ACCENT};'>❓ Step 3: Ask Your Question</h3>", unsafe_allow_html=True)

    query = st.text_input(
        "Enter your question:",
        "What are recurring employee concerns?"
    )

    top_k = st.slider(
        "Number of feedback chunks to retrieve:",
        min_value=10,
        max_value=200,
        value=40,
        step=10
    )

    # ---------- RUN RAG ----------
    if st.button("🔍 Generate Summary"):

        with st.spinner("Retrieving relevant feedback..."):
            retrieved = retrieve(df_chunks, index, query, top_k)

        with st.expander("📑 View retrieved feedback used for summary"):
            for _, row in retrieved.iterrows():
                st.markdown(f"""
                <div style="
                    background:{CARD_BG};
                    padding:8px;
                    margin:5px;
                    border-radius:8px;
                    border-left:5px solid {BORDER};
                    color:{WHITE};
                ">
                    {row['chunk']}
                </div>
                """, unsafe_allow_html=True)

        prompt = build_prompt(query, retrieved["chunk"].tolist())

        with st.spinner("🧠 Generating sentiment summary..."):
            result = call_gemini(prompt)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---------- OUTPUT ----------
        st.markdown(f"<h3 style='color:{ACCENT};'>🎯 Sentiment Summary</h3>", unsafe_allow_html=True)

        sentiment = result.get("overall_sentiment", "Not available")
        summary = result.get("summary", "No summary generated.")

        # sentiment card
        st.markdown(f"""
            <div style="
                background:{CARD_BG};
                padding:16px;
                border-radius:10px;
                border-left:8px solid {ACCENT};
                color:{WHITE};
                margin-bottom:20px;
            ">
                <h3>🧭 Overall Sentiment:
                    <span style="color:{ACCENT};">{sentiment}</span>
                </h3>
            </div>
        """, unsafe_allow_html=True)

        # summary card
        st.markdown(f"""
            <div style="
                background:{ANSWER_BG};
                padding:20px;
                border-radius:10px;
                border-left:8px solid {ACCENT};
                color:{WHITE};
            ">
                <h3>📝 Summary of Employee Concerns</h3>
                <p style="font-size:16px; line-height:1.6;">
                    {summary}
                </p>
            </div>
        """, unsafe_allow_html=True)

else:
    st.warning("⬆ Upload your CSV to begin.")

# ------------------ FOOTER ------------------
st.markdown(f"""
<div style='text-align:center; margin-top:35px; color:{WHITE}; opacity:0.6;'>
    Built with Streamlit • FAISS • SentenceTransformers • Gemini (RAG)
</div>
""", unsafe_allow_html=True)
