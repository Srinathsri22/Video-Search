import streamlit as st
import os
import json
import whisper
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load models
whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths
UPLOAD_DIR = "uploads"
JSON_PATH = "storage/chunked_transcripts.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("storage", exist_ok=True)

# --- UTILS ---
def transcribe_and_chunk(video_path, chunk_duration=30):
    result = whisper_model.transcribe(video_path, verbose=False)
    chunks = []
    video_name = os.path.basename(video_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for segment in result.get("segments", []):
        start = segment["start"]
        text = segment["text"].strip()

        chunks.append({
            "video_name": video_name,
            "timestamp": timestamp,
            "start_time": f"{int(start//60):02}:{int(start%60):02}",
            "text": text
        })

    return chunks

def save_chunks(chunks):
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(chunks)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)

def load_chunks():
    if not os.path.exists(JSON_PATH):
        return []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_chunk_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search_chunks(query, chunks, index, threshold=1.0):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), len(chunks))

    results_by_video = {}

    for idx, dist in zip(I[0], D[0]):
        if dist <= threshold:
            entry = chunks[idx]
            video_name = entry["video_name"]
            if video_name not in results_by_video:
                results_by_video[video_name] = {
                    "video_name": video_name,
                "timestamp": entry["timestamp"],
                "matches": []
                }
            results_by_video[video_name]["matches"].append({
                "start_time": entry["start_time"],
                "text": entry["text"],
                "score": float(dist)
            })

    return list(results_by_video.values())

# --- UI ---
st.set_page_config(page_title="🎥 Video Search", layout="centered")
st.title("🎥 AI Video Search Engine (By Video)")

# Upload
st.header("📤 Upload a Video")
uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)
    st.success("✅ Video uploaded")

    if st.button("📝 Transcribe & Chunk"):
        with st.spinner("Transcribing and chunking..."):
            chunks = transcribe_and_chunk(video_path)
            save_chunks(chunks)
        st.success("✅ Transcript chunks saved!")

# Search
st.header("🔍 Search Across Videos")
query = st.text_input("Type a question or keyword...")

if query:
    chunks = load_chunks()
    if not chunks:
        st.warning("No transcripts available.")
    else:
        index = build_chunk_index(chunks)
        results = search_chunks(query, chunks, index, threshold=1.0)

        if not results:
            st.warning("❌ No relevant videos found.")
        else:
            st.subheader("📌 Matching Videos")
            for i, r in enumerate(results, 1):
                st.markdown(f"**{i}. 📽️ Video:** `{r['video_name']}`  \n🗓️ Uploaded at: `{r['timestamp']}`")
                for match in r["matches"]:
                    st.markdown(f"- ⏱️ `{match['start_time']}` — {match['text']}")
                st.markdown("---")
