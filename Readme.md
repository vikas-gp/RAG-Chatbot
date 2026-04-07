# Retrieval-Augmented Generation (RAG) Chatbot Project

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers user queries strictly based on a given textbook:

**"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"**

The project is implemented in two approaches:

1. Manual RAG Pipeline (Pure Python)
2. LangChain-based Automated RAG Pipeline

The goal of this project is to understand both low-level implementation and high-level abstraction of RAG systems.

---

## 1. Manual RAG Implementation (Python)

### Pipeline Overview

The manual RAG system is built step-by-step without using frameworks.

#### 1. Data Ingestion

* PDF loaded using `pdfplumber`
* Text extracted page-wise

#### 2. Text Splitting

* Text divided into chunks (~400–500 tokens)
* Total chunks generated: ~4000+

#### 3. Embeddings

* Model: `sentence-transformers/all-MiniLM-L6-v2`
* Converts text chunks into 384-dimensional vectors

#### 4. Vector Database

* FAISS used for similarity search
* Stored:

  * `index.faiss`
  * `chunks.pkl`

#### 5. Retrieval

* Query converted to embedding
* FAISS used to retrieve top-k relevant chunks

#### 6. Generation

* Groq API used with LLaMA models
* Strict prompt:

  * Answer only from context
  * No hallucination
  * If not found → return fallback message

#### 7. UI (Streamlit)

* Chat interface with:

  * User and bot messages
  * Typing animation
  * Scrollable chat
* Displays:

  * Answer
  * Source chunks with page numbers

#### 8. Explainability

* Extracts:

  * Chapter
  * Page number
* Displays source context for transparency

---

## 2. LangChain RAG Implementation

The same pipeline is implemented using LangChain to automate and modularize the workflow.

### Pipeline Mapping

| Manual Step    | LangChain Component            |
| -------------- | ------------------------------ |
| PDF Loading    | PyPDFLoader                    |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings     | HuggingFaceEmbeddings          |
| Vector DB      | FAISS                          |
| Retrieval      | Retriever                      |
| Prompting      | ChatPromptTemplate             |
| LLM            | Custom Groq Wrapper            |
| Pipeline       | LCEL (Runnable chain)          |

---

### Summary

* Manual RAG provides full control and transparency
* LangChain reduces boilerplate and improves modularity
* Trade-off: Control vs Abstraction

---

## Setup Instructions

### 1. Clone Repository

```
git clone <repo-url>
cd <project-folder>
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Add Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

## Running the Project

### Run Streamlit App

```
streamlit run app.py
```

---

## Key Learnings

* Understanding of complete RAG pipeline
* Difference between embeddings and LLMs
* Role of vector databases in semantic search
* Importance of prompt design in RAG systems
* LangChain abstractions and LCEL workflow
* Debugging real-world RAG issues

---

## Conclusion

This project demonstrates both low-level and high-level implementations of a RAG system. It highlights the importance of retrieval quality, prompt design, and modular architecture in building reliable AI application.
