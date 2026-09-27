import streamlit as st
import base64
import fitz
import pickle
import os
import sys
from pathlib import Path


# ============================================================
#                 PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTOR_PATH     = BASE_DIR / "vectorizer.pkl"
MODEL_PATH      = BASE_DIR / "legal_model.pkl"
BACKGROUND_PATH = BASE_DIR / "assets" / "img.png"


# ============================================================
#                 CHECK REQUIRED FILES
# ============================================================

if not VECTOR_PATH.exists():
    st.error(f"❌ Vectorizer file not found:\n{VECTOR_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"❌ ML model file not found:\n{MODEL_PATH}")
    st.stop()

if not BACKGROUND_PATH.exists():
    st.warning(f"⚠️ Background image not found:\n{BACKGROUND_PATH}")


# ============================================================
#                 ADD PROJECT ROOT TO PATH
# ============================================================

PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
#                 CORE MODULES
# ============================================================

from core import preprocess, summarizer, ner
from core.clause_extractor import extract_clauses


# ============================================================
#                 AI MODEL (HuggingFace)
# ============================================================

from transformers import pipeline


@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    return pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name
    )


nlp_model = load_model()


def ask_ai(question, context):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    response = nlp_model(prompt, max_length=300, num_return_sequences=1)
    return response[0]["generated_text"]


# ============================================================
#                 LOAD ML MODEL + VECTORIZER
# ============================================================

with open(VECTOR_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ============================================================
#                 STREAMLIT MAIN UI SETTINGS
# ============================================================

st.set_page_config(
    page_title="⚖️ Legal Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.85)),
            url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: white;
            font-family: 'Poppins', 'Segoe UI', sans-serif;
        }}

        /* Hide streamlit default header/footer chrome */
        header[data-testid="stHeader"] {{
            background: transparent;
        }}
        #MainMenu, footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True
    )


if BACKGROUND_PATH.exists():
    set_background(BACKGROUND_PATH)


# ============================================================
#                 FULL CUSTOM STYLING
# ============================================================

st.markdown("""
<style>

/* ---------------- Global ---------------- */
html, body, [class*="css"] {
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* ---------------- Scrollbar ---------------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.3); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00adb5, #0077b6);
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover { background: #00e0ff; }

/* ---------------- Main Title ---------------- */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 0;
    background: linear-gradient(90deg, #ffffff, #00e0ff, #0077b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 18px rgba(0, 224, 255, 0.55));
    letter-spacing: 0.5px;
}

.sub-title {
    text-align: center;
    color: #d6e4f0;
    font-size: 15px;
    margin-top: 4px;
    margin-bottom: 24px;
    opacity: 0.9;
}

/* ---------------- Tabs ---------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(0, 0, 0, 0.35);
    padding: 8px;
    border-radius: 14px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #cfe6ee;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.25s ease;
    border: 1px solid transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(0, 173, 181, 0.15);
    color: white;
    border: 1px solid rgba(0, 224, 255, 0.3);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #00adb5, #0077b6) !important;
    color: white !important;
    box-shadow: 0 0 20px rgba(0, 173, 181, 0.5);
    border: 1px solid rgba(0, 224, 255, 0.5) !important;
}

/* Hide the ugly tab underline */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}

/* ---------------- Section Headers ---------------- */
h2, h3 {
    color: #eaf6ff !important;
    font-weight: 700 !important;
    text-shadow: 0 0 12px rgba(0, 224, 255, 0.25);
}

h3 {
    margin-top: 22px !important;
    font-size: 20px !important;
    border-left: 4px solid #00e0ff;
    padding-left: 12px;
}

/* ---------------- Pill Labels ---------------- */
.pill-label {
    display: inline-block;
    padding: 12px 24px;
    border-radius: 30px;
    font-size: 16px;
    font-weight: 700;
    color: white;
    margin-top: 14px;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #00adb5, #0077b6);
    box-shadow: 0 6px 20px rgba(0, 173, 181, 0.35),
                inset 0 1px 0 rgba(255,255,255,0.2);
    letter-spacing: 0.3px;
}

/* ---------------- File Uploader (FIXED) ---------------- */
[data-testid="stFileUploader"] {
    width: 100%;
}

[data-testid="stFileUploader"] > section,
[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(10, 20, 30, 0.65) !important;
    border: 2px dashed rgba(0, 224, 255, 0.45) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    transition: all 0.25s ease;
    backdrop-filter: blur(8px);
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #00e0ff !important;
    background-color: rgba(0, 173, 181, 0.15) !important;
    box-shadow: 0 0 25px rgba(0, 224, 255, 0.25);
}

/* All text inside uploader -> white */
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileName"] span {
    color: #ffffff !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] svg {
    fill: #00e0ff !important;
    color: #00e0ff !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #b8d6e0 !important;
}

/* "Browse files" button inside uploader */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, #00adb5, #0077b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    transition: all 0.2s ease;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(90deg, #00c8d1, #0090d6) !important;
    box-shadow: 0 0 18px rgba(0, 224, 255, 0.5);
    transform: translateY(-1px);
}

/* Uploaded file chip */
[data-testid="stFileUploaderFile"] {
    background-color: rgba(0, 173, 181, 0.18) !important;
    border: 1px solid rgba(0, 224, 255, 0.35) !important;
    border-radius: 10px !important;
    padding: 6px 12px !important;
    margin-top: 8px;
}

/* Delete (X) icon on the file chip */
[data-testid="stFileUploaderFile"] svg {
    fill: #ff6b6b !important;
}

/* ---------------- Text Area & Text Input (FIXED) ---------------- */
.stTextArea textarea,
.stTextInput input {
    background-color: rgba(10, 20, 30, 0.65) !important;
    color: #ffffff !important;
    border: 1.5px solid rgba(0, 224, 255, 0.35) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    font-size: 15px !important;
    transition: all 0.25s ease;
    backdrop-filter: blur(8px);
}

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #a8c4d0 !important;
    opacity: 0.85;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #00e0ff !important;
    box-shadow: 0 0 0 3px rgba(0, 224, 255, 0.18),
                0 0 20px rgba(0, 224, 255, 0.25) !important;
    outline: none !important;
}

/* ---------------- Buttons ---------------- */
div.stButton > button {
    background: linear-gradient(90deg, #00adb5, #0077b6) !important;
    color: white !important;
    padding: 10px 26px !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    border: 1px solid rgba(0, 224, 255, 0.4) !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 16px rgba(0, 173, 181, 0.35);
    letter-spacing: 0.3px;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #00c8d1, #0090d6) !important;
    box-shadow: 0 6px 24px rgba(0, 224, 255, 0.55) !important;
    transform: translateY(-2px);
    border: 1px solid #00e0ff !important;
}

div.stButton > button:active {
    transform: translateY(0px);
}

/* ---------------- Result Cards ---------------- */
.result-box,
.summary-box,
.entity-box,
.clause-box,
.ai-answer {
    color: #ffffff !important;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.65), rgba(0, 40, 60, 0.55)) !important;
    padding: 18px 22px;
    border-radius: 14px;
    font-size: 16px;
    margin-top: 12px;
    line-height: 1.7;
    border: 1px solid rgba(0, 224, 255, 0.25);
    border-left: 4px solid #00e0ff;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
    word-wrap: break-word;
}

.result-box b { color: #00e0ff; }

/* Clause box slight variation */
.clause-box {
    border-left: 4px solid #00adb5;
    margin-bottom: 12px;
}

/* ---------------- Alerts (success/warning/info) ---------------- */
.stAlert {
    background-color: rgba(10, 20, 30, 0.75) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 224, 255, 0.25) !important;
    backdrop-filter: blur(8px);
}
.stAlert * { color: white !important; }

/* ---------------- Expander ---------------- */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
    background-color: rgba(10, 20, 30, 0.6) !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0, 224, 255, 0.2) !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: rgba(0, 173, 181, 0.2) !important;
}
[data-testid="stExpander"] * { color: white !important; }

/* ---------------- Spinner ---------------- */
.stSpinner > div {
    border-top-color: #00e0ff !important;
}

/* ---------------- Footer ---------------- */
.footer-text {
    text-align: center;
    margin-top: 50px;
    padding: 18px;
    color: #b8d6e0;
    font-size: 14px;
    border-top: 1px solid rgba(0, 224, 255, 0.15);
    letter-spacing: 0.4px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
#                      TITLE
# ============================================================

st.markdown(
    "<h1 class='main-title'>⚖️ Legal Document Intelligence System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p class='sub-title'>AI-powered tool for "
    "<b>document classification • clause extraction • summarization • entity recognition</b></p>",
    unsafe_allow_html=True
)


# ============================================================
#                      TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📄 Document Analysis",
    "📑 Clause Extraction",
    "🤖 AI Legal Assistant"
])


# ============================================================
#                 PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(file):
    text = ""
    file.seek(0)
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text


# ============================================================
#                 TAB 1 – DOCUMENT ANALYSIS
# ============================================================

with tab1:
    st.header("📄 Document Type Detection, Summary & Entities")

    st.markdown("<div class='pill-label'>📄 Upload PDF</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], key="t1_pdf", label_visibility="collapsed")

    st.markdown("<div class='pill-label'>✏️ Or paste your document text</div>", unsafe_allow_html=True)
    text_input = st.text_area("", height=180, label_visibility="collapsed",
                              placeholder="Paste your legal document text here...")

    if st.button("🔍 Analyze Document"):
        text = extract_text_from_pdf(uploaded_file) if uploaded_file else text_input

        if text.strip() == "":
            st.warning("Please upload a file or enter text.")
        else:
            with st.spinner("Processing..."):
                cleaned = preprocess.clean_text(text)
                tokens = preprocess.tokenize_text(cleaned)
                filtered = preprocess.remove_stopwords(tokens)
                lemmatized = preprocess.lemmatize_tokens(filtered)

                final_text = " ".join(lemmatized)
                X = vectorizer.transform([final_text])
                pred = model.predict(X)[0]

                st.success("Document Analyzed Successfully!")
                st.markdown(
                    f"<div class='result-box'>📌 Document Type: <b>{pred}</b></div>",
                    unsafe_allow_html=True
                )

                st.write("### 🧾 Summary")
                st.markdown(
                    f"<div class='summary-box'>{summarizer.summarize_text(text)}</div>",
                    unsafe_allow_html=True
                )

                st.write("### 🔍 Extracted Entities")
                ents = ner.extract_entities(text)
                if ents:
                    for e in ents:
                        st.markdown(
                            f"<div class='entity-box'>🏷️ <b>{e['label']}</b> → {e['text']}</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No entities detected.")

                with st.expander("📜 View Extracted Text"):
                    st.write(text)


# ============================================================
#                 TAB 2 – CLAUSE EXTRACTION
# ============================================================

with tab2:
    st.header("📑 Clause Extraction")

    st.markdown("<div class='pill-label'>📄 Upload PDF</div>", unsafe_allow_html=True)
    uploaded_clause = st.file_uploader("", type=["pdf"], key="t2_pdf", label_visibility="collapsed")

    if st.button("🔍 Extract Clauses"):
        if uploaded_clause:
            text = extract_text_from_pdf(uploaded_clause)
            clauses = extract_clauses(text)

            if clauses:
                st.success(f"✅ {len(clauses)} Clauses Extracted Successfully!")
                for i, c in enumerate(clauses, 1):
                    st.markdown(
                        f"<div class='clause-box'><b>Clause {i}:</b> {c}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("No clauses found.")
        else:
            st.warning("Please upload a PDF.")


# ============================================================
#                 TAB 3 – AI LEGAL ASSISTANT
# ============================================================

with tab3:
    st.header("🤖 Ask AI About Your Document")

    st.markdown("<div class='pill-label'>📄 Upload PDF</div>", unsafe_allow_html=True)
    uploaded_ai = st.file_uploader("", type=["pdf"], key="t3_pdf", label_visibility="collapsed")

    st.markdown("<div class='pill-label'>✏️ Or paste your document text</div>", unsafe_allow_html=True)
    text_ai = st.text_area("", height=150, label_visibility="collapsed",
                           placeholder="Paste your legal document text here...")

    st.markdown("<div class='pill-label'>❓ Ask your question</div>", unsafe_allow_html=True)
    question = st.text_input("", key="question_input", label_visibility="collapsed",
                             placeholder="e.g. What is the termination clause in this agreement?")

    if st.button("🧠 Get Answer"):
        context = extract_text_from_pdf(uploaded_ai) if uploaded_ai else text_ai

        if context.strip() == "":
            st.warning("Please provide a document or text.")
        elif question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("AI is generating your answer..."):
                response = ask_ai(question, context)
                st.markdown(
                    f"<div class='ai-answer'>💡 {response}</div>",
                    unsafe_allow_html=True
                )


# ============================================================
#                      FOOTER
# ============================================================

st.markdown(
    "<p class='footer-text'>⚖️ LegalDoc AI — Making legal document analysis simple and effortless 📄🤖</p>",
    unsafe_allow_html=True
)