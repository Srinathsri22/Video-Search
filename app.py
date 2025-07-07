import streamlit as st
import os
import json
import whisper
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load Whisper + Embedding model
whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths
UPLOAD_DIR = "uploads"
JSON_PATH = "storage/transcripts.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("storage", exist_ok=True)

# ---------- Utilities ----------
def transcribe_video(video_path):
    result = whisper_model.transcribe(video_path, verbose=False)
    return result["text"].strip()

def save_transcription(video_name, text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "video_name": video_name,
        "transcription": text,
        "timestamp": timestamp
    }

    # Append to existing JSON
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
    else:
        existing = []

    existing.append(new_entry)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)

def load_transcriptions():
    if not os.path.exists(JSON_PATH):
        return []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(data):
    texts = [entry["transcription"] for entry in data]
    embeddings = embed_model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

def search(query, data, index, k=1):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k)

    results = []
    for idx in I[0]:
        entry = data[idx]
        results.append({
            "video_name": entry["video_name"],
            "timestamp": entry["timestamp"],
            "text_snippet": entry["transcription"][:200] + "..."
        })
    return results

# ---------- Streamlit App ----------
st.set_page_config(page_title="🎥 Video Search Engine", layout="centered")
st.title("🎥 Video Search Using FAISS & SentenceTransformers")

st.header("📤 Upload a Video")
uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)
    st.success("✅ Video uploaded successfully")

    if st.button("📝 Transcribe with Whisper"):
        with st.spinner("Transcribing..."):
            full_text = transcribe_video(video_path)
            save_transcription(uploaded_file.name, full_text)
        st.success("✅ Transcription saved!")

# ---------- Search Interface ----------
st.header("🔎 Search Transcriptions")
query = st.text_input("Type a question or keyword...")

if query:
    data = load_transcriptions()
    if not data:
        st.warning("No transcripts available.")
    else:
        index, _ = build_index(data)
        results = search(query, data, index, k=1)

        st.subheader("📌 Top Match")
        for r in results:
            st.markdown(f"**📽️ Video:** `{r['video_name']}`")
            st.markdown(f"**⏱️ Uploaded:** `{r['timestamp']}`")
            st.markdown(f"**🧠 Snippet:** {r['text_snippet']}")
