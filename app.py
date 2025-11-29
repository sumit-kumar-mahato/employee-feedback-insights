import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================================
# 🔐 Gemini API Key
GEMINI_API_KEY = "AIzaSyBO7n0YCDBgyUH22_u0Q_8FGPPYsvEOmik"
# ==========================================================

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Employee Feedback Insights — Sentiment Summary (RAG)",
    page_icon="📘",
    layout="wide"
)

# ALWAYS VISIBLE COLORS
WHITE = "#FFFFFF"
CARD_BG = "rgba(255,255,255,0.12)"
ANSWER_BG = "rgba(255,255,255,0.20)"
ACCENT = "#00D16C"
BORDER = "#00AEEF"

# ------------------ FIXED HEADER (FINAL) ------------------
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

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# ------------------ CHUNK TEXT ------------------
def chunk_text(text, chunk_size=250, overlap=40):
    words = text.split()
    if not words:
        return []
    chunks = []
    pos = 0
    while pos < len(words):
        chunk = " ".join(words[pos : pos + chunk_size])
        chunks.append(chunk)
        pos += chunk_size - overlap
    return chunks

# ------------------ FAISS INDEX ------------------
def build_index(texts):
    vectors = model.encode(texts.tolist(), convert_to_numpy=True, show_progress_bar=False)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def retrieve(df_local, index, query, k):
    k = min(k, index.ntotal)
    q_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, k)
    results = df_local.iloc[I[0]].copy()
    results["distance"] = D[0]
    return results

# ------------------ SHORT SUMMARY PROMPT ------------------
def build_prompt(query, feedback):

    examples = "\n".join([f"- {t}" for t in feedback])

    return f"""
You are an expert HR analyst.

Using ONLY the retrieved employee feedback below, write a **short sentiment summary** (4–6 lines).
Do NOT use bullet points.
Do NOT write long essays.
Write a tight, concise narrative summary.

QUESTION:
{query}

RETRIEVED FEEDBACK:
{examples}

Return ONLY this JSON:

{{
    "overall_sentiment": "Positive / Negative / Mixed",
    "summary": "A short paragraph summarizing the main employee concerns."
}}
"""

# ------------------ CALL GEMINI ------------------
def call_gemini(prompt):

    if not GEMINI_API_KEY:
        return {"error": "Missing API key"}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except:
        return {"error": "Invalid JSON", "raw": raw}


# ==========================================================
# ------------------ FILE UPLOAD ---------------------------
# ==========================================================
st.markdown(f"<h3 style='color:{ACCENT};'>📂 Step 1: Upload CSV</h3>", unsafe_allow_html=True)
st.info("The CSV must contain a **feedback_text** column.")

uploaded = st.file_uploader("", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    if "feedback_text" not in df.columns:
        st.error("❌ Missing 'feedback_text' column.")
        st.stop()

    st.success(f"✔ Loaded {len(df)} feedback entries.")
    st.dataframe(df.head())

    st.markdown("<hr>", unsafe_allow_html=True)

    # ------------------ CHUNKING ------------------
    st.markdown(f"<h3 style='color:{ACCENT};'>🧩 Step 2: Chunking</h3>", unsafe_allow_html=True)

    rows = []
    for text in df["feedback_text"]:
        for ch in chunk_text(str(text)):
            rows.append({"chunk": ch})

    df_chunks = pd.DataFrame(rows)
    st.success(f"Created **{len(df_chunks)} chunks**")

    # ------------------ INDEX ------------------
    with st.spinner("🔧 Building FAISS index..."):
        index = build_index(df_chunks["chunk"])

    st.markdown("<hr>", unsafe_allow_html=True)

    # ------------------ QUESTION ------------------
    st.markdown(f"<h3 style='color:{ACCENT};'>❓ Step 3: Ask Question</h3>", unsafe_allow_html=True)

    query = st.text_input(
        "Enter your question:",
        "What are recurring employee concerns?"
    )

    top_k = st.slider("Number of chunks to retrieve:", 10, 200, 40, step=10)

    # ------------------ RUN RAG ------------------
    if st.button("🔍 Generate Summary"):

        with st.spinner("Retrieving relevant feedback..."):
            retrieved = retrieve(df_chunks, index, query, top_k)

        # Optional: View retrieved
        with st.expander("📑 View retrieved chunks"):
            for _, row in retrieved.iterrows():
                st.markdown(f"""
                <div style="
                    background:{CARD_BG};
                    padding:10px;
                    margin:6px;
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

        # ------------------ FINAL OUTPUT ------------------
        st.markdown(f"<h3 style='color:{ACCENT};'>🎯 Sentiment Summary</h3>", unsafe_allow_html=True)

        sentiment = result.get("overall_sentiment", "Not available")
        summary = result.get("summary", "")

        # Sentiment card
        st.markdown(f"""
            <div style="
                background:{CARD_BG};
                padding:16px;
                border-radius:10px;
                border-left:8px solid {ACCENT};
                color:{WHITE};
                margin-bottom:20px;
            ">
                <h3>🧭 Overall Sentiment: <span style="color:{ACCENT};">{sentiment}</span></h3>
            </div>
        """, unsafe_allow_html=True)

        # Summary card
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
    Built with Streamlit • FAISS • SentenceTransformers • Gemini RAG
</div>
""", unsafe_allow_html=True)
