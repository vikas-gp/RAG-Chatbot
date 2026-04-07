import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, index, chunks, k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0] if len(chunks[i]) > 200]

    return results