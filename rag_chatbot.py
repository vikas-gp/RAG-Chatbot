import pickle
from vector_store import load_index
from retriever import retrieve
from llm import generate_answer

index = load_index()

with open("vector_db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


if __name__ == "__main__":
    print("\nRAG Chatbot Ready (type exit to quit)\n")

    while True:
        query = input("Enter your Query:").strip()
       
        if query.lower() in ["exit", "quit", "bye"]:
            print("exiting chatbot..")
            break

        retrieved_chunks = retrieve(query, index, chunks, k=5)

        context = "\n\n".join(retrieved_chunks[:3])

        answer = generate_answer(query, context)

        print('\nAnswer:\n', answer)
        print('\n'+ '-'*50 + '\n')