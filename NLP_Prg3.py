# Install gensim if not already installed
# !pip install gensim==4.3.2

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# --- Step 1: Load Dataset ---
with open("data_1000.txt", "r", encoding="utf-8", errors="ignore") as f:
    sentences = [simple_preprocess(line) for line in f if line.strip()]

# --- Step 2: Train Word2Vec Model ---
model = Word2Vec(
    sentences,
    vector_size=100,  # Size of word vectors
    window=3,         # Context window size
    min_count=1,      # Ignore rare words
    sg=1,             # Skip-gram model
    epochs=200        # Number of training passes
)

# --- Step 3: Interactive Query Loop ---
while True:
    word = input("\nEnter a word (or 'exit' to quit): ").strip().lower()
    if word == "exit":
        break
    if word in model.wv:
        print(f"\nSimilar words to '{word}':")
        for w, score in model.wv.most_similar(word, topn=5):
            print(f"{w:12s} -> {score:.3f}")
    else:
        print(f"'{word}' not found in vocabulary.")
