"""Streamlit interface for the trained SMS/email spam detector."""

from pathlib import Path

import joblib
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = PROJECT_DIR / "model" / "tfidf_vectorizer.pkl"

st.set_page_config(
    page_title="SMS/Email Spam Detector",
    page_icon="📩",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    .stApp { background: linear-gradient(145deg, #f4f8f5 0%, #edf3ef 55%, #f7f6ef 100%); }
    .block-container { max-width: 850px; padding-top: 3rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #153d32; }
    p, label, button, div { font-family: 'DM Sans', sans-serif; }
    .intro { color: #4c6259; font-size: 1.05rem; line-height: 1.65; }
    .eyebrow { color: #267a5d; font-weight: 700; font-size: .76rem; text-transform: uppercase; }
    .flow { border-top: 1px solid #d5e1d9; margin-top: 1.5rem; padding-top: 1.1rem; color: #496158; }
    div.stButton > button { background: #176b50; color: white; border: 0; border-radius: 7px; min-height: 3rem; font-weight: 700; }
    div.stButton > button:hover { background: #10533e; color: white; border: 0; }
    [data-testid="stTextArea"] textarea { border-color: #c6d7cd; border-radius: 7px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            "Model files are missing. Open a terminal in the project folder and run "
            "`.\\.venv\\Scripts\\python.exe train_model.py` first."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


st.markdown('<div class="eyebrow">A simple machine learning project</div>', unsafe_allow_html=True)
st.title("📩 SMS/Email Spam Detector")
st.markdown(
    '<p class="intro">Paste a message below to check whether it looks like spam. '
    'The classifier uses TF-IDF features and a Multinomial Naive Bayes model.</p>',
    unsafe_allow_html=True,
)

try:
    model, vectorizer = load_artifacts()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

message = st.text_area(
    "Message to check",
    height=210,
    placeholder="Paste or type an SMS or email message here...",
    max_chars=10000,
)

if st.button("Detect Spam", type="primary", use_container_width=True):
    if not isinstance(message, str) or not message.strip():
        st.warning("Enter a message before selecting Detect Spam.")
    else:
        try:
            features = vectorizer.transform([message.strip()])
            prediction = int(model.predict(features)[0])
            spam_probability = float(model.predict_proba(features)[0][list(model.classes_).index(1)])
            if prediction == 1:
                st.error("🚨 SPAM", icon="🚨")
                st.write(f"Spam likelihood: **{spam_probability:.1%}**")
            else:
                st.success("✅ NOT SPAM (HAM)", icon="✅")
                st.write(f"Spam likelihood: **{spam_probability:.1%}**")
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            st.error(f"This message could not be checked. Please try another message. ({error})")

st.markdown(
    """
    <div class="flow">
      <strong>How it works</strong><br>
      Input Text &nbsp;→&nbsp; TF-IDF &nbsp;→&nbsp; Machine Learning Model &nbsp;→&nbsp; Spam / Not Spam
    </div>
    """,
    unsafe_allow_html=True,
)