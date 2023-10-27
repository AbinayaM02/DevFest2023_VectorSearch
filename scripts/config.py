MAX_DOCS = 100000 # higher the value, longer will be the time to get embeddings
TOP_K = 5 # set it based on your need

# Data and model used for Cohere API
WIKI_EMB = f"Cohere/wikipedia-22-12-en-embeddings"
WIKI_EMB_MODEL = f"multilingual-22-12"

# Data and model used for offline search
WIKI_DATA = f"Cohere/wikipedia-22-12"
EMB_MODEL = f"BAAI/bge-large-en-v1.5"

# Path to store the embeddings and the index
EMB_PATH = f"../data/wiki_embeddings_offline.npy"
FAISS_PATH = f"../data/faiss_embeddings"