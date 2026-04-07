from pdf_loader import load_pdf
from text_splitter import split_text
from embeddings import create_embeddings, model
from vector_store import create_faiss_index, search_index, save_index
import pickle
import os


def build_pipeline(pdf_path):
   
    text = load_pdf(pdf_path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    index = create_faiss_index(embeddings)

    os.makedirs("vector_db", exist_ok=True)
   

    save_index(index)

    with open("vector_db/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    
    
    return chunks, index


if __name__ == "__main__":
    pdf_path = "data/book.pdf"   # make sure this path is correct

    chunks, index = build_pipeline(pdf_path)