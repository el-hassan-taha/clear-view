import streamlit as st
import re
import pickle
import json
import numpy as np
import os
import nltk
from nltk.corpus import stopwords

# For lazy loading of TF models
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Clear-View: News Verifier", page_icon="🛡️", layout="centered")

# --- Premium Glassmorphism CSS injected ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Force main app background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460) !important;
        background-attachment: fixed !important;
        color: #e2e8f0 !important;
    }
    
    /* Also set header to transparent */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Force Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 52, 96, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Text in sidebar */
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    
    /* Exception for dropdown text */
    div[role="listbox"] * {
        color: #1e293b !important;
    }

    /* Titles */
    h1 {
        text-align: center !important;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 0.2rem !important;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }

    /* Glassmorphic Text Area */
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(4px) !important;
        transition: all 0.3s ease-in-out !important;
    }

    /* Placeholder text */
    .stTextArea > div > div > textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }

    /* Focus text area */
    .stTextArea > div > div > textarea:focus {
        border-color: #4facfe !important;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.3) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }

    /* Vibrant Gradient Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.5) !important;
        color: white !important;
    }

    /* Glowing Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
        border-radius: 10px !important;
    }
    
    /* Make Alert text visible */
    .stAlert p {
        color: #f8fafc !important;
        font-weight: 500 !important;
    }
    
    /* Base markdown text white in main container */
    [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Preprocessing Function
def preprocess_input(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Load Metadata
@st.cache_data
def load_metadata():
    if os.path.exists("models/metadata.json"):
        with open("models/metadata.json", "r") as f:
            return json.load(f)
    return {"logistic": 0.0, "lstm": 0.0}

metadata = load_metadata()

# Lazy Loaders
@st.cache_resource
def load_logistic_model():
    with open('models/logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_resource
def load_lstm_model():
    from tensorflow.keras.models import load_model
    model = load_model('models/lstm_model.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# --- Sidebar ---
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "LSTM Neural Network"]
)

if model_choice == "Logistic Regression":
    st.sidebar.metric(label="Training Accuracy", value=f"{metadata.get('logistic', 0) * 100:.2f}%")
else:
    st.sidebar.metric(label="Training Accuracy", value=f"{metadata.get('lstm', 0) * 100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color:#cbd5e1; font-size:0.9rem;'>The Logistic Regression model uses TF-IDF features. The LSTM model uses word embeddings and captures sequential context.</p>", unsafe_allow_html=True)

# --- Main Interface Layout ---
st.title("🛡️ Clear-View: News Verifier")
st.markdown("<div class='subtitle'>Professional Fact-Checking Dashboard. Paste news text below.</div>", unsafe_allow_html=True)

news_input = st.text_area("Article Content:", height=250, placeholder="Paste the news article text here...")

# Centered Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_button = st.button("Run Prediction")

if run_button:
    if not news_input.strip():
        st.warning("Please enter some text to predict.")
    else:
        with st.spinner(f"Analyzing with {model_choice}..."):
            try:
                cleaned_text = preprocess_input(news_input)
                
                if model_choice == "Logistic Regression":
                    model, vectorizer = load_logistic_model()
                    transformed_text = vectorizer.transform([cleaned_text])
                    prediction_prob = model.predict_proba(transformed_text)[0]
                    # LogReg predict_proba returns [prob_fake, prob_real] since 0=Fake, 1=Real
                    prob_real = prediction_prob[1]
                else:
                    model, tokenizer = load_lstm_model()
                    from tensorflow.keras.preprocessing.sequence import pad_sequences
                    sequence = tokenizer.texts_to_sequences([cleaned_text])
                    padded_sequence = pad_sequences(sequence, maxlen=100)
                    prob_real = float(model.predict(padded_sequence)[0][0])
                
                # Output Display
                st.markdown("### Prediction Results")
                
                # We show columns for the status and probability
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    if prob_real > 0.5:
                        st.success("✅ Confidence High: This news appears REAL.")
                    else:
                        st.error("🚨 Warning: This news is flagged as FAKE.")
                
                with res_col2:
                    st.markdown(f"**Truth Probability:** {prob_real:.2%}")
                    st.progress(prob_real)
            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}\n\nMake sure the models are trained and saved.")
