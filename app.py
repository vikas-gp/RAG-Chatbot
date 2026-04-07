import streamlit as st
import pickle
from vector_store import load_index
from retriever import retrieve
from llm import generate_answer
import time
import re

def extract_metadata(text):

    
    chapter_match = re.search(r"Chapter\s+\d+", text)
    chapter = chapter_match.group(0) if chapter_match else "Unknown Chapter"

    
    numbers = re.findall(r"\b\d{2,4}\b", text)
    page = numbers[-1] if numbers else "N/A"

    return chapter, page

st.set_page_config(page_title="RAG Chatbot", layout="wide")

col1, col2, col3  = st.columns([1,6,1])
with col2:
    st.title("AI Textbook Assistant ")


@st.cache_resource
def load_resources():
    index = load_index()
    with open("vector_db/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

index, chunks = load_resources()



if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False



col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    chat_container = st.container(height=600)

with chat_container:
    st.markdown("""
    <style>
        .user_msg{
            text-align: right;
            background-color: #2b313e;
            padding: 10px 14px;
            border-radius:12px;
            margin: 8px 0;
            display: inline-block;
            max-width:60%;
        }
        .bot_msg{
            text-align: left;
            background-color: #1f2937;
            padding: 10px 14px;
            border-radius:12px;
            margin: 8px 0;
            display: inline-block;
            max-width:60%;  
        }
        .msg_container{
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="msg_container" style="text-align:right;">
                    <div class="user_msg">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="msg_container">
                    <div class="bot_msg">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True
            )
            
            if "sources" in msg:
                with st.expander("Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        chapter, page = extract_metadata(src)
                        st.markdown(
                            f"""
                            **{i}. {chapter}**
                            ***----***
                            {src[:250]}...
                            <div style="text-align:right; font-size:14px; color:gray;">
                            Page: {page}
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("---")

    st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <script>
        const bottom = window.parent.document.getElementById("bottom");
        if (bottom) {
            bottom.scrollIntoView({behavior: "smooth"});
        }
    </script>
    """,
    unsafe_allow_html=True
)

with col2:
    input_col1, input_col2 = st.columns([6, 1])

    with input_col1:

        if st.session_state.clear_input:
            st.session_state.user_input = ""
            st.session_state.clear_input = False

        query = st.text_input(
            "",
            placeholder="Type your question",
            key="user_input"
        )

    with input_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        ask = st.button("Ask")


if ask or (query and query != st.session_state.last_query):

    st.session_state.last_query = query

    if query.strip():

        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        # Retrieval
        retrieved_chunks = retrieve(query, index, chunks, k=5)
        context = "\n\n".join(retrieved_chunks[:3])

        # LLM Response
        with st.spinner("Thinking..."):
            response = generate_answer(query, context)

        
        # Add bot response
        with chat_container:
            placeholder = st.empty()
            full_text =""

            for char in response:
                full_text += char
                placeholder.markdown(
                    f"""
                    <div class="msg_container">
                        <div class="bot_msg">{full_text}</div>
                    </div>
                    """, unsafe_allow_html=True
                )
                time.sleep(0.01)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "sources": retrieved_chunks[:3]}
        )

        # Clear input safely
        st.session_state.clear_input = True

        st.rerun()