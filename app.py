import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import google.generativeai as genai
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted
import shutil

# --- Aggressive cache clear (keep this for dev if previous issues persist) ---
st.cache_resource.clear()
# --- End aggressive cache clear ---

# 1. Load Environment Variables from .env file
load_dotenv()

# Get the Google API Key from the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
else:
    genai.configure(api_key=GOOGLE_API_KEY)


# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                st.warning(f"Could not extract text from page {page_num + 1} of {pdf.name}. It might be an image-based page.")

    # --- TEMPORARY DEBUG PRINT (Comment out or remove this block after debugging text extraction) ---
    # st.markdown("---")
    # st.subheader("Raw Extracted Text (for debugging)")
    # st.text_area("Review this to see what the AI is actually reading.", text, height=400, key="raw_text_debug")
    # st.markdown("--- End of Raw Extracted Text ---")
    # -----------------------------
    
    return text

def get_text_chunks(text):
    """Splits a long text into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_embedding_model():
    """Initializes and caches the Google Generative AI Embeddings model."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def get_vector_store(text_chunks=None, persist_directory="./chroma_db"):
    """
    Creates or loads a vector store from text chunks using Google Generative AI embeddings.
    The embeddings are saved to disk in a directory named 'chroma_db'.
    """
    embeddings = get_embedding_model()

    if os.path.exists(persist_directory) and text_chunks is None:
        st.info("Loading existing vector store...")
        vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    elif text_chunks:
        st.info("Creating new vector store from documents...")
        vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=persist_directory)
        vector_store.persist()
    else:
        st.warning("No text chunks provided and no existing vector store to load. Please upload documents.")
        return None
    return vector_store


@st.cache_resource
def get_conversational_chain(_vector_store):
    """
    Creates a conversational retrieval chain using a Gemini Pro model.
    Caches the chain for performance.
    """
    if _vector_store is None:
        st.error("Vector store is not available. Please process documents first.")
        return None

    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.5)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(),
        return_source_documents=True,
        output_key="answer"
    )
    return conversation_chain

def handle_user_input(user_question):
    """Handles user questions and displays the AI response along with source documents."""
    if st.session_state.conversation:
        try:
            # Add user question to chat history for display immediately
            if st.session_state.chat_history is None:
                st.session_state.chat_history = []
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            # Display all previous messages + current user message (this re-renders all messages up to current)
            for message_item in st.session_state.chat_history:
                if message_item["role"] == "user":
                    st.chat_message("user").write(message_item["content"])
                else:
                    st.chat_message("ai").write(message_item["content"])

            # Add a loading spinner/message for AI response
            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    # *** CRITICAL FIX: Prepare chat history as a list of (user_message, ai_message) tuples ***
                    chat_history_for_chain = []
                    # Iterate through chat_history in pairs of (user, ai) messages
                    # We need to ensure we have a full pair before adding to history for the chain
                    for i in range(0, len(st.session_state.chat_history) - 1, 2):
                        user_msg_content = st.session_state.chat_history[i]["content"]
                        ai_msg_content = st.session_state.chat_history[i+1]["content"]
                        chat_history_for_chain.append((user_msg_content, ai_msg_content))

                    response = st.session_state.conversation({
                        'question': user_question,
                        'chat_history': chat_history_for_chain
                    })

                    ai_answer = response.get('answer', 'No answer found.')
                    st.write(ai_answer)

            # Append AI message after generation
            st.session_state.chat_history.append({"role": "ai", "content": ai_answer})

            # Display source documents
            if 'source_documents' in response and response['source_documents']:
                with st.expander("📚 See Sources"):
                    for i, doc in enumerate(response['source_documents']):
                        st.markdown(f"**Source {i+1}:**")
                        page_content = doc.page_content if doc.page_content else "No content preview available."
                        st.markdown(f"```text\n{page_content[:500]}...\n```")
                        if doc.metadata:
                            st.write(f"Metadata: {doc.metadata}")
                        st.markdown("---")

        except ResourceExhausted:
            st.error("Quota Exceeded: You've made too many requests. Please wait a minute or two and try again. For sustained usage, consider enabling billing on your Google Cloud project.")
        except Exception as e:
            st.error(f"An error occurred during AI interaction: {e}")
            st.info("Please try processing documents again, or restart the app if the issue persists.")
    else:
        st.warning("Please process documents first before asking questions.")

def clear_chat_history():
    """Clears the chat history and resets the conversation chain."""
    st.session_state.chat_history = None
    st.session_state.conversation = None
    st.session_state.vector_store = None
    st.rerun()

def delete_chroma_db_and_restart():
    """Deletes the chroma_db folder and restarts the Streamlit app."""
    if os.path.exists("./chroma_db"):
        st.info("Deleting existing 'chroma_db' folder...")
        try:
            shutil.rmtree("./chroma_db")
            st.success("Successfully deleted 'chroma_db'. Restarting app...")
            st.session_state.chat_history = None
            st.session_state.conversation = None
            st.session_state.vector_store = None
            st.rerun()
        except OSError as e:
            st.error(f"Error deleting 'chroma_db': {e}. Please ensure no files are open in it.")
    else:
        st.info("No 'chroma_db' folder found to delete.")
        st.session_state.chat_history = None
        st.session_state.conversation = None
        st.session_state.vector_store = None
        st.rerun()


# --- Streamlit UI ---

st.set_page_config(page_title="Intelligent Document QA System", layout="wide")
st.title("Intelligent Document QA System 🤖")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for document upload and controls
with st.sidebar:
    st.header("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDF Files and Click on the Process Button",
        accept_multiple_files=True,
        type=["pdf"]
    )

    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                st.session_state.vector_store = get_vector_store(text_chunks)
                if st.session_state.vector_store:
                    st.session_state.conversation = get_conversational_chain(st.session_state.vector_store)
                    st.success(f"Processed {len(pdf_docs)} PDF(s) and created {len(text_chunks)} text chunks! Embeddings stored. Ready for questions! 🎉")
                    st.session_state.chat_history = None
                else:
                    st.error("Failed to create or load vector store. Please check logs.")
        else:
            st.warning("Please upload at least one PDF file!")

    st.markdown("---")
    st.header("App Controls")
    if st.button("Clear Chat History", on_click=clear_chat_history):
        pass

    if st.button("Delete Vector Store & Restart", on_click=delete_chroma_db_and_restart):
        pass

# Main content area
st.markdown("---")
st.write("Welcome! Upload PDFs in the sidebar and ask questions here.")

# Load existing vector store and conversation chain if available on app start
if st.session_state.vector_store is None and os.path.exists("./chroma_db"):
    st.session_state.vector_store = get_vector_store(text_chunks=None)
    if st.session_state.vector_store:
        st.session_state.conversation = get_conversational_chain(st.session_state.vector_store)
        st.success("Loaded existing documents from 'chroma_db'. Ready for questions! ✅")
        # Display existing chat history if it was reloaded (e.g., from future persistence)
        if st.session_state.chat_history:
            for message_item in st.session_state.chat_history:
                if message_item["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message_item["content"])
                else:
                    with st.chat_message("ai"):
                        st.write(message_item["content"])

# User question input
user_question = st.chat_input("Ask a Question about the Documents:")

if user_question:
    handle_user_input(user_question)