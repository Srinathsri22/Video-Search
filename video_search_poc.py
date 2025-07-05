# video_search_poc.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Simulated transcript chunks (video_id, start_time, text)
transcript_chunks = [
    {
        "video_id": "Video1",
        "start_time": "00:00:00",
        "text": "Welcome to the dashboard tutorial. We'll show you how to get started with setting up your widgets."
    },
    {
        "video_id": "Video1",
        "start_time": "00:01:30",
        "text": "To change the theme of the dashboard, go to settings and select your preferred color scheme."
    },
    {
        "video_id": "Video2",
        "start_time": "00:00:10",
        "text": "To install the application, download the setup file and run the installer with admin permissions."
    },
    {
        "video_id": "Video2",
        "start_time": "00:01:00",
        "text": "If you face issues during installation, disable your antivirus temporarily and try again."
    },
    {
        "video_id": "Video3",
        "start_time": "00:00:20",
        "text": "An effective marketing strategy starts with understanding your target audience and their pain points."
    },
    {
        "video_id": "Video3",
        "start_time": "00:02:00",
        "text": "Using social media platforms like Instagram and LinkedIn can greatly boost brand visibility."
    },
    {
        "video_id": "Video4",
        "start_time": "00:00:45",
        "text": "You can upload your sales data in Excel format and generate insights in the analytics dashboard."
    },
    {
        "video_id": "Video4",
        "start_time": "00:01:50",
        "text": "Comparing quarterly performance metrics helps in spotting growth trends and revenue dips."
    },
    {
        "video_id": "Video5",
        "start_time": "00:00:15",
        "text": "Always use a strong password and enable two-factor authentication to keep your account secure."
    },
    {
        "video_id": "Video5",
        "start_time": "00:01:10",
        "text": "If you've forgotten your password, use the 'Forgot Password' option to reset it securely."
    }
]


# Step 2: Load Sentence Transformer model
print("🔄 Loading sentence-transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Create embeddings for all transcript chunks
texts = [chunk["text"] for chunk in transcript_chunks]
print("🔄 Generating embeddings...")
embeddings = model.encode(texts)

# Step 4: Build FAISS index
print("🔄 Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 5: Search function
def search_query(query, k=1):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k)

    results = []
    for idx in I[0]:
        match = transcript_chunks[idx]
        results.append({
            "video_id": match["video_id"],
            "start_time": match["start_time"],
            "text": match["text"]
        })
    return results

# Step 6: Try it with user input
if __name__ == "__main__":
    print("\n🎯 Video Search Engine (FAISS + Sentence Transformers)")
    while True:
        query = input("\n🔎 Enter your search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        top_results = search_query(query, k=1)
        print("\n📌 Top Match:")
        for result in top_results:
            print(f"📽️ Video: {result['video_id']}")
            print(f"⏱️ Time: {result['start_time']}")
            print(f"🧠 Text: {result['text']}")
