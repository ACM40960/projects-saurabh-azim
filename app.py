# ============================================================
# MediGuide - Flask Application
# AI-Powered Analysis for Intelligent Healthcare Assistance
# ============================================================
#
# This file is the main backend application for MediGuide.
#
# Main responsibilities:
# 1. Load environment variables
# 2. Initialize Hugging Face embeddings
# 3. Connect to the existing Pinecone vector index
# 4. Configure document retrieval
# 5. Configure the GPT-4o language model
# 6. Build the Retrieval-Augmented Generation (RAG) pipeline
# 7. Serve the MediGuide chat interface
# 8. Process user questions through the /get endpoint
#
# Authors:
# Saurabh Kumbhar - 25204974
# Azim Hassan   - 25203062
# ============================================================


# ----------------------------
# Standard Library Imports
# ----------------------------

import os


# ----------------------------
# Flask Imports
# ----------------------------

from flask import Flask, render_template, request


# ----------------------------
# Environment Configuration
# ----------------------------

from dotenv import load_dotenv


# ----------------------------
# LangChain Imports
# ----------------------------

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate


# ----------------------------
# MediGuide Project Imports
# ----------------------------

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


# ============================================================
# 1. Initialize Flask Application
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. Load Environment Variables
# ============================================================

# Load API keys from the .env file.
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Validate Pinecone API key.
if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing. "
        "Please add it to the .env file."
    )


# Validate OpenAI API key.
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Please add it to the .env file."
    )


# ============================================================
# 3. Load Hugging Face Embeddings
# ============================================================

# Load the same embedding model that was used when
# the medical knowledge base was indexed in Pinecone.
#
# Model:
# sentence-transformers/all-MiniLM-L6-v2
#
# Vector dimension:
# 384
embeddings = download_hugging_face_embeddings()


# ============================================================
# 4. Connect to Existing Pinecone Index
# ============================================================

# IMPORTANT:
# The index name must exactly match the index
# created in your Pinecone indexing script.
INDEX_NAME = "mediguide"


# Connect to the existing Pinecone vector store.
#
# This does not upload the PDF documents again.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)


# ============================================================
# 5. Configure Medical Document Retriever
# ============================================================

# Retrieve the top 3 most relevant medical text chunks
# for every user question.
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
    },
)


# ============================================================
# 6. Configure GPT-4o
# ============================================================

# GPT-4o generates the final response using the
# medical context retrieved from Pinecone.
#
# temperature=0:
# Keeps responses focused and consistent for
# question-answering tasks.
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ============================================================
# 7. Configure MediGuide Prompt
# ============================================================

# The system prompt is stored in:
#
# src/prompt.py
#
# LangChain automatically inserts retrieved document
# content into the {context} placeholder.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            "{input}",
        ),
    ]
)


# ============================================================
# 8. Build Question-Answering Chain
# ============================================================

# Combine:
#
# Retrieved medical documents
#        +
# MediGuide system prompt
#        +
# User question
#        ↓
# GPT-4o
#
question_answer_chain = create_stuff_documents_chain(
    chat_model,
    prompt,
)


# ============================================================
# 9. Build Complete RAG Pipeline
# ============================================================

# Complete MediGuide workflow:
#
# User Question
#       ↓
# Hugging Face Query Embedding
#       ↓
# Pinecone Similarity Search
#       ↓
# Top 3 Relevant Medical Chunks
#       ↓
# MediGuide System Prompt
#       ↓
# GPT-4o
#       ↓
# Final Response
#
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain,
)


# ============================================================
# 10. Home Route
# ============================================================

@app.route("/")
def index():
    """
    Render the main MediGuide chatbot interface.
    """

    return render_template("chat.html")


# ============================================================
# 11. Chat Route
# ============================================================

@app.route("/get", methods=["POST"])
def chat():
    """
    Receive the user's healthcare question,
    process it through the RAG pipeline,
    and return the generated response.
    """

    # Retrieve the message sent from the frontend.
    user_message = request.form.get(
        "msg",
        "",
    ).strip()


    # Prevent empty questions.
    if not user_message:
        return (
            "Please enter a valid healthcare question.",
            400,
        )


    try:

        # ----------------------------------------------------
        # Log User Question
        # ----------------------------------------------------

        print(
            f"\n[MediGuide] User Question: {user_message}"
        )


        # ----------------------------------------------------
        # Run RAG Pipeline
        # ----------------------------------------------------

        response = rag_chain.invoke(
            {
                "input": user_message,
            }
        )


        # ----------------------------------------------------
        # Extract Generated Answer
        # ----------------------------------------------------

        answer = response.get(
            "answer",
            "I could not generate a response.",
        )


        # Log the generated response.
        print(
            f"[MediGuide] Response: {answer}\n"
        )


        # Return the answer to the web interface.
        return str(answer)


    except Exception as error:

        # ----------------------------------------------------
        # Development Error Logging
        # ----------------------------------------------------

        print(
            f"\n[MediGuide] Error: {error}\n"
        )


        # Return a clean message to the user instead
        # of exposing the Python traceback.
        return (
            "MediGuide is currently unable to process "
            "your request. Please try again shortly.",
            500,
        )


# ============================================================
# 12. Run MediGuide Locally
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=8080,
        debug=True,
    )

