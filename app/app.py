"""
LokGeet MVP Streamlit app (app/app.py)

Features:
- Audio upload / record (browser)
- ASR via faster-whisper (local or HF)
- Show transcript, transliteration, translation placeholders
- Simple metadata form + consent checkbox
- Export as JSON / CSV
"""

import streamlit as st
from pathlib import Path
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Optional

# ASR libs - runtime import for graceful fallback
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# Transliteration helper (optional)
try:
    # package: indic-transliteration
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLIT_AVAILABLE = True
except Exception:
    TRANSLIT_AVAILABLE = False

# App paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = BASE_DIR / "data.json"
MODEL = None

st.set_page_config(page_title="LokGeet — Collect Folk Songs", layout="wide")
st.markdown(
    """
    <style>
    /* Modern color palette */
    :root {
        --primary: #2c786c;
        --primary-light: #429e8f;
        --secondary: #f5b461;
        --accent: #fae3c6;
        --bg-main: #fcfaf7;
        --text-color: #2c786c;
        --text-light: #429e8f;
    }

    /* Main layout */
    .main, .stApp {
        background: linear-gradient(135deg, var(--bg-main) 0%, #f7efe8 100%) !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, var(--primary) 0%, var(--primary-light) 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 0.5em 1.5em !important;
        border: none !important;
        box-shadow: 0 2px 12px rgba(44, 120, 108, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(44, 120, 108, 0.3) !important;
    }

    /* Input fields */
    .stTextInput>div>div>input {
        background-color: white !important;
        border: 2px solid var(--accent) !important;
        border-radius: 10px !important;
        color: var(--text-dark) !important;
        font-size: 1.1em !important;
        padding: 1em !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-light) !important;
        box-shadow: 0 0 0 2px rgba(66, 158, 143, 0.2) !important;
    }

    /* Text areas */
    .stTextArea>div>div>textarea {
        background-color: white !important;
        border: 2px solid var(--accent) !important;
        border-radius: 12px !important;
        color: var(--text-dark) !important;
        font-size: 1.15em !important;
        line-height: 1.6 !important;
        padding: 1em !important;
        transition: all 0.3s ease !important;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: var(--primary-light) !important;
        box-shadow: 0 0 0 2px rgba(66, 158, 143, 0.2) !important;
    }

    /* Headings and text */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stMarkdown p, 
    .stMarkdown span, .stMarkdown div, div[data-testid="stText"] {
        color: var(--text-color) !important;
    }
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5em !important;
    }
    h3 {
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin: 1em 0 !important;
    }
    .stMarkdown p {
        font-size: 1.1em !important;
        line-height: 1.6 !important;
    }
    /* All text inputs and text areas text color */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, 
    .stSelectbox>div>div {
        color: var(--text-color) !important;
    }

    /* Sidebar */
    .stSidebar {
        background: linear-gradient(180deg, var(--bg-main) 0%, #f0f7f5 100%) !important;
        border-left: 1px solid rgba(44, 120, 108, 0.1) !important;
        padding: 2em 1em !important;
    }

    /* File uploader */
    .stUploadButton>button {
        background-color: var(--accent) !important;
        color: var(--primary) !important;
        font-weight: 600 !important;
        border: 2px dashed var(--primary-light) !important;
        border-radius: 12px !important;
        padding: 1.5em !important;
    }

    /* Select boxes */
    .stSelectbox>div>div {
        background-color: white !important;
        border: 2px solid var(--accent) !important;
        border-radius: 10px !important;
    }

    /* Info boxes */
    .stInfo {
        background-color: rgba(66, 158, 143, 0.1) !important;
        border-left: 4px solid var(--primary) !important;
        padding: 1em !important;
        border-radius: 0 8px 8px 0 !important;
    }

    /* Success messages */
    .stSuccess {
        background-color: rgba(66, 158, 143, 0.1) !important;
        border-left: 4px solid var(--primary) !important;
        color: var(--primary) !important;
        padding: 1em !important;
        border-radius: 0 8px 8px 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helpers ----------
def load_model(model_size="small"):
    global MODEL
    if not WHISPER_AVAILABLE:
        return None
    if MODEL is None:
        # Options: "tiny", "base", "small", "medium", "large"
        MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
    return MODEL

def transcribe_with_whisper(audio_path: str, model_size="small"):
    model = load_model(model_size)
    if model is None:
        return {"transcript": "", "segments": [], "language": None, "error": "faster-whisper not available"}
    segments, info = model.transcribe(audio_path, beam_size=5)
    transcript = " ".join([s.text.strip() for s in segments])
    return {"transcript": transcript, "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments], "language": info.language}

def romanize(text: str, lang_code: str="hi"):
    if not TRANSLIT_AVAILABLE:
        return ""
    # Map some example lang codes -> sanscript scheme; adjust as needed
    lang_to_scheme = {"hi": sanscript.DEVANAGARI, "mr": sanscript.DEVANAGARI, "bn": sanscript.BENGALI, "ta": sanscript.TAMIL, "te": sanscript.TELUGU}
    src = lang_to_scheme.get(lang_code, sanscript.DEVANAGARI)
    return transliterate(text, src, sanscript.ITRANS)

def load_db() -> list:
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text(encoding="utf8"))
    return []

def save_entry(entry: Dict[str, Any]):
    db = load_db()
    db.append(entry)
    DB_FILE.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf8")

# ---------- UI ----------

st.markdown(
    """
    <h1 style='color:#b77b2b; font-family:Georgia,serif; text-align:center;'>LokGeet <span style='font-size:0.6em;'>&mdash; Field Recorder & Folk Song Collector</span></h1>
    <p style='text-align:center; color:#444; font-size:1.1em;'>Upload or record a folk song / lullaby audio.<br>The assistant will transcribe, transliterate, and help you add metadata & consent.</p>
    """,
    unsafe_allow_html=True
)


st.markdown("---")
col1, col2 = st.columns([2,1], gap="large")


with col1:
    st.markdown("<h3 style='color:#b77b2b;'>1. Upload or Record Audio</h3>", unsafe_allow_html=True)
    audio_file = st.file_uploader("🎵 Upload audio file (wav, mp3, m4a, ogg)", type=["wav","mp3","m4a","ogg"])
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    if st.button("🎤 Record from microphone (experimental)"):
        st.info("Browser recording is experimental — try uploading for now.")
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    st.markdown("<b>ASR model settings</b>", unsafe_allow_html=True)
    model_choice = st.selectbox("Whisper model (local)", ["small","base","tiny"], index=0)
    st.info("If <code>faster-whisper</code> isn't installed, transcription will be disabled and you'll be asked to paste text manually.", icon="ℹ️")

with col2:
    st.markdown("<h3 style='color:#b77b2b;'>Quick Actions</h3>", unsafe_allow_html=True)
    if st.button("📖 Show saved entries"):
        db = load_db()
        if db:
            st.json(db)
        else:
            st.info("No entries saved yet.")


st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)
process = st.button("✨ Process audio") if audio_file else False

if audio_file and process:
    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)
    # Save to temp file
    suffix = Path(audio_file.name).suffix
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    saved_path = UPLOAD_DIR / f"{ts}{suffix}"
    with open(saved_path, "wb") as f:
        f.write(audio_file.getbuffer())
    st.success(f"✅ Saved to {saved_path}")

    with st.spinner("📝 Running ASR (speech-to-text)..."):
        asr_result = transcribe_with_whisper(str(saved_path), model_choice)
    if "error" in asr_result:
        st.error("❌ ASR unavailable: " + asr_result["error"])
        st.info("Please paste transcript manually or install faster-whisper.")
        transcript = st.text_area("Transcript (paste or edit)", value="", height=120)
    else:
        transcript = st.text_area("Transcript (edit to correct)", value=asr_result.get("transcript",""), height=120)
        st.markdown(f"<b>Detected language:</b> <span style='color:#b77b2b'>{asr_result.get('language','')}</span> &nbsp; <b>Segments:</b> {len(asr_result.get('segments',[]))}", unsafe_allow_html=True)

    # Transliteration

    st.markdown("<h3 style='color:#b77b2b;'>2. Transliteration & Translation</h3>", unsafe_allow_html=True)
    detected_lang = asr_result.get("language") if asr_result.get("language") else st.text_input("Detected Language (ISO code)", value="hi")
    transliteration = ""
    if transcript.strip():
        transliteration = romanize(transcript, lang_code=detected_lang) if TRANSLIT_AVAILABLE else ""
    translit_box = st.text_area("Romanized transliteration", value=transliteration, height=100)
    translation = st.text_area("Quick English translation (manual for MVP)", value="", height=100)


    st.markdown("<h3 style='color:#b77b2b;'>3. Metadata & Consent</h3>", unsafe_allow_html=True)
    with st.form("meta"):
        title = st.text_input("Title / short description", value="")
        performer = st.text_input("Performer / contributor (anonymize if requested)", value="")
        location = st.text_input("Location (village, district, state)", value="")
        context = st.text_input("Context (e.g., lullaby, harvest song, wedding song)", value="")
        date_of_recording = st.date_input("Date of recording", value=datetime.utcnow().date())
        consent = st.checkbox("I confirm I have obtained consent and the user agreed to upload this recording (required)")

        submitted = st.form_submit_button("💾 Save entry")
        if submitted:
            if not consent:
                st.error("Consent is required to save entries.")
            else:
                entry = {
                    "id": ts,
                    "title": title,
                    "performer": performer,
                    "location": location,
                    "context": context,
                    "date_of_recording": str(date_of_recording),
                    "uploaded_at": ts,
                    "audio_path": str(saved_path),
                    "transcript": transcript,
                    "transliteration": translit_box,
                    "translation": translation,
                    "detected_language": detected_lang
                }
                save_entry(entry)
                st.success("✅ Saved! You can export entries from the 'Export' panel.")


# Export panel
st.sidebar.header("📤 Export & Admin")
if st.sidebar.button("Export all as JSON"):
    db = load_db()
    out = BASE_DIR / "export.json"
    out.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf8")
    st.sidebar.success(f"✅ Written {len(db)} entries to {out}")

if st.sidebar.button("Download example CSV"):
    st.sidebar.info("Use the exported JSON for now. CSV export coming soon.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span style='color:#b77b2b'><b>Developer notes:</b></span> install <code>faster-whisper</code> for ASR and <code>indic-transliteration</code> for transliteration support.",
    unsafe_allow_html=True
)
