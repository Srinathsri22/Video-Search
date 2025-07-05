# Video-Search
# 🎥 Video Transcript Search - PoC

This is a Python-based proof-of-concept for fast semantic search over video transcript chunks using FAISS and Sentence Transformers.

## 🚀 What It Does
- Stores transcript chunks from videos along with timestamps
- Converts those chunks into semantic embeddings
- Uses FAISS for fast vector similarity search
- On a query, returns the most relevant video + timestamp

## 📦 Tech Stack
- [sentence-transformers](https://www.sbert.net/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- Python 3.7+

---

## 🧪 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
