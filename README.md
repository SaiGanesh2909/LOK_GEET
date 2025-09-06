# LokGeet — Open-Source Folk Song Collector (MVP)

LokGeet is an open-source assistant & toolkit to collect, transcribe, transliterate, and preserve folk songs and lullabies across Indic languages.

## Quick start (development)
1. Clone repo
2. Create a venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install:
   ```bash
   pip install -r app/requirements.txt
   ```
4. Run Streamlit:
   ```bash
   cd app
   streamlit run app.py
   ```

## Notes
- For on-device ASR, install `faster-whisper`. If not available, use manual transcript input.
- The `data.json` file in `app/` stores saved entries (for dev). Use S3/DB for production.
- The `system_prompt.txt` contains the system prompt to use if you later wire LokGeet to a chat LLM (Hugging Face / Dify / self-hosted LLM).

## Next steps (recommended)
- Replace local JSON with Postgres + S3.
- Add an editing UI for segment-level ASR corrections.
- Integrate translation model or RAG to surface related songs in corpus.
