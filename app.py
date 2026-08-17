# ============================================================
# MediGuide - Flask Application
# AI-Powered Analysis for Intelligent Healthcare Assistance
# ============================================================
#
# This module acts as the main backend application for MediGuide.
#
# Core responsibilities:
# 1. Load environment variables and API credentials
# 2. Initialize the medical text embedding model
# 3. Connect to the existing Pinecone vector database
# 4. Configure semantic retrieval
# 5. Configure GPT-4o for response generation
# 6. Enable history-aware retrieval for follow-up questions
# 7. Maintain temporary conversational memory
# 8. Process user questions through the RAG pipeline
# 9. Clear conversation history when requested
# 10. Provide a lightweight health-check endpoint
#
# Authors:
# Saurabh Kumbhar - 25204974
# Azim Hassan   - 25203062
# ============================================================


# ------------------------------------------------------------
# Standard Library Imports
# ------------------------------------------------------------

import os


# ------------------------------------------------------------
# Flask Imports
# ------------------------------------------------------------

from flask import Flask, render_template, request


# ------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------

from dotenv import load_dotenv


# ------------------------------------------------------------
# LangChain / AI Imports
# ------------------------------------------------------------

from langchain_openai import ChatOpenAI

from langchain_pinecone import PineconeVectorStore

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)


# ------------------------------------------------------------
# MediGuide Project Imports
# ------------------------------------------------------------

from src.helper import (
    download_hugging_face_embeddings,
)

from src.prompt import (
    contextualize_q_prompt,
    qa_prompt,
)


# ============================================================
# 1. Initialize Flask Application
# ============================================================

# Create the Flask application instance.
#
# Flask automatically looks for:
# - HTML templates inside templates/
# - CSS / JavaScript / assets inside static/
app = Flask(__name__)


# ============================================================
# 2. Load Environment Variables
# ============================================================

# Load values from the local .env file.
#
# Expected variables:
# OPENAI_API_KEY
# PINECONE_API_KEY
load_dotenv()


# Retrieve the OpenAI API key.
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)


# Retrieve the Pinecone API key.
PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)


# Stop the application early if the OpenAI key
# has not been configured correctly.
if not OPENAI_API_KEY:

    raise ValueError(
        "OPENAI_API_KEY is missing from the .env file."
    )


# Stop the application early if the Pinecone key
# has not been configured correctly.
if not PINECONE_API_KEY:

    raise ValueError(
        "PINECONE_API_KEY is missing from the .env file."
    )


# ============================================================
# 3. Initialize Hugging Face Embeddings
# ============================================================

# Load the embedding model defined in src/helper.py.
#
# MediGuide uses the same embedding model for:
# - indexing medical document chunks
# - converting user queries into semantic vectors
#
# The current model is:
# sentence-transformers/all-MiniLM-L6-v2
#
# This model produces 384-dimensional vectors.
embeddings = download_hugging_face_embeddings()


# ============================================================
# 4. Connect to the Existing Pinecone Index
# ============================================================

# Name of the Pinecone index created by store_index.py.
#
# This must match the name used while indexing the
# medical knowledge base.
INDEX_NAME = "mediguide"


# Connect LangChain to the existing Pinecone vector store.
#
# This does NOT re-upload or re-index the PDF documents.
# It only creates a connection to the vectors already stored.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)


# ============================================================
# 5. Configure the Base Retriever
# ============================================================

# Convert the Pinecone vector store into a retriever.
#
# search_type="similarity":
# retrieves documents based on semantic similarity.
#
# k=3:
# returns the three most relevant medical text chunks.
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
    },
)


# ============================================================
# 6. Configure GPT-4o
# ============================================================

# GPT-4o is used for two important tasks:
#
# 1. Rewriting context-dependent follow-up questions
#    into standalone questions.
#
# 2. Generating the final medical response using the
#    retrieved Pinecone context.
#
# temperature=0 keeps responses more focused
# and consistent for question-answering tasks.
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ============================================================
# 7. Create the History-Aware Retriever
# ============================================================

# A standard retriever only sees the current question.
#
# Example:
#
# User:
# "What is acne?"
#
# Follow-up:
# "How can it be treated?"
#
# The follow-up question alone does not mention acne.
#
# The history-aware retriever uses the previous conversation
# to reformulate the question internally as:
#
# "How can acne be treated?"
#
# This improved standalone question is then used for
# Pinecone semantic search.
history_aware_retriever = create_history_aware_retriever(
    chat_model,
    retriever,
    contextualize_q_prompt,
)


# ============================================================
# 8. Create the Question-Answering Chain
# ============================================================

# This chain combines:
#
# - retrieved medical context
# - conversation history
# - the latest user question
# - MediGuide's QA system prompt
#
# The complete information is then sent to GPT-4o.
question_answer_chain = create_stuff_documents_chain(
    chat_model,
    qa_prompt,
)


# ============================================================
# 9. Build the Conversational RAG Pipeline
# ============================================================

# Complete MediGuide pipeline:
#
# User Question
#      ↓
# Conversation History
#      ↓
# Follow-Up Question Reformulation
#      ↓
# Query Embedding
#      ↓
# Pinecone Similarity Search
#      ↓
# Top 3 Relevant Medical Chunks
#      ↓
# MediGuide QA Prompt
#      ↓
# GPT-4o
#      ↓
# Final Response
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)


# ============================================================
# 10. Temporary Conversation Memory
# ============================================================

# Store messages from the current conversation.
#
# Example structure:
#
# [
#     HumanMessage(content="What is acne?"),
#     AIMessage(content="Acne is ..."),
#     HumanMessage(content="How can it be treated?"),
#     AIMessage(content="Acne can be treated ...")
# ]
#
# Important:
# This is in-memory storage.
#
# It works well for:
# - local development
# - testing
# - single-user demonstrations
#
# It is NOT suitable for a multi-user production deployment
# because all users would share the same global memory.
chat_history = []


# ============================================================
# 11. Home Route
# ============================================================

@app.route("/")
def home():
    """
    Render the main MediGuide web interface.

    Flask loads chat.html from the templates/ directory.
    """

    return render_template(
        "chat.html"
    )


# ============================================================
# 12. Chat Route
# ============================================================

@app.route(
    "/get",
    methods=["POST"]
)
def get_response():
    """
    Process a user's healthcare question.

    Workflow:
    1. Read the question from the frontend.
    2. Pass the question and chat history into the RAG chain.
    3. Retrieve relevant medical context from Pinecone.
    4. Generate a GPT-4o response.
    5. Save the question and answer to conversation memory.
    6. Return the answer to the frontend.
    """

    global chat_history


    # --------------------------------------------------------
    # Read User Input
    # --------------------------------------------------------

    # Extract the "msg" form field sent by chat.html.
    # .strip() removes leading and trailing whitespace.
    user_message = request.form.get(
        "msg",
        "",
    ).strip()


    # Reject empty questions before calling the AI pipeline.
    if not user_message:

        return (
            "Please enter a valid healthcare question.",
            400,
        )


    try:

        # ----------------------------------------------------
        # Development Logging
        # ----------------------------------------------------

        print(
            f"\n[MediGuide] User: {user_message}"
        )


        # ----------------------------------------------------
        # Run Conversational RAG
        # ----------------------------------------------------

        # Pass both the latest question and the previous
        # conversation to the history-aware RAG chain.
        response = rag_chain.invoke(
            {
                "input": user_message,
                "chat_history": chat_history,
            }
        )


        # ----------------------------------------------------
        # Extract the Generated Answer
        # ----------------------------------------------------

        # The retrieval chain returns a dictionary.
        # "answer" contains GPT-4o's final response.
        answer = response.get(
            "answer",
            "I could not generate a response.",
        )


        # ----------------------------------------------------
        # Save Conversation to Memory
        # ----------------------------------------------------

        # Store the user's latest question.
        chat_history.append(
            HumanMessage(
                content=user_message
            )
        )


        # Store MediGuide's generated answer.
        chat_history.append(
            AIMessage(
                content=answer
            )
        )


        # ----------------------------------------------------
        # Limit Conversation Memory
        # ----------------------------------------------------

        # Keep only the most recent 12 messages.
        #
        # Each conversation turn produces:
        # - 1 HumanMessage
        # - 1 AIMessage
        #
        # Therefore, 12 messages represent approximately
        # the most recent 6 question-answer exchanges.
        #
        # This prevents the prompt from growing indefinitely.
        if len(chat_history) > 12:

            chat_history = chat_history[-12:]


        # ----------------------------------------------------
        # Development Logging
        # ----------------------------------------------------

        print(
            f"[MediGuide] Assistant: {answer}"
        )


        print(
            f"[MediGuide] Memory size: "
            f"{len(chat_history)} messages\n"
        )


        # Return the generated answer to chat.html.
        return str(answer)


    except Exception as error:

        # ----------------------------------------------------
        # Error Handling
        # ----------------------------------------------------

        # Print the actual error in the terminal for debugging.
        print(
            f"[MediGuide] Error: {error}"
        )


        # Return a user-friendly message instead of
        # exposing internal Python errors in the interface.
        return (
            "Sorry, MediGuide is currently unable to "
            "process your question. Please try again.",
            500,
        )


# ============================================================
# 13. Clear Conversation Memory
# ============================================================

@app.route(
    "/clear",
    methods=["POST"]
)
def clear_conversation():
    """
    Clear MediGuide's current in-memory conversation.

    The frontend should call this endpoint when the user
    presses the Clear Conversation button.
    """

    global chat_history


    # Reset the conversation history.
    chat_history = []


    # Development log.
    print(
        "[MediGuide] Conversation memory cleared."
    )


    return (
        "Conversation cleared.",
        200,
    )


# ============================================================
# 14. Health Check Endpoint
# ============================================================

@app.route("/health")
def health():
    """
    Return a lightweight status response.

    This endpoint is useful for:
    - checking whether Flask is running
    - validating deployment health
    - checking the active model and index
    - inspecting the current memory size
    """

    return {
        "application": "MediGuide",
        "status": "healthy",
        "model": "gpt-4o",
        "index": INDEX_NAME,
        "memory_messages": len(chat_history),
    }


# ============================================================
# 15. Run Flask Development Server
# ============================================================

if __name__ == "__main__":

    # Start the application locally.
    #
    # URL:
    # http://127.0.0.1:8080
    #
    # debug=True is useful during local development because
    # Flask automatically reloads when source files change.
    #
    # Disable debug mode for production deployment.
    app.run(
        host="127.0.0.1",
        port=8080,
        debug=True,
    )
