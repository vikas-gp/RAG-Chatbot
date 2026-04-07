import faiss
import numpy as np

def create_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index

def search_index(index, query_embedding, chunks, top_k=3):
    query_embedding = np.array([query_embedding]).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = [chunks[i] for i in indices[0]]

    return results

def save_index(index, path="vector_db/faiss_index.index"):
    faiss.write_index(index,path)

def load_index(path="vector_db/faiss_index.index"):
    return faiss.read_index(path)