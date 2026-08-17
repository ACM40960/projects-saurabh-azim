# ============================================================
# MediGuide - Flask Application
# AI-Powered Analysis for Intelligent Healthcare Assistance
# ============================================================
#
# Main responsibilities:
# 1. Load environment variables
# 2. Connect to Pinecone
# 3. Configure Hugging Face embeddings
# 4. Configure GPT-4o
# 5. Create history-aware retrieval
# 6. Maintain conversation memory
# 7. Process medical questions
# 8. Clear conversation history
#
# Authors:
# Saurabh Kumbhar - 25204974
# Azim Hassan   - 25203062
# ============================================================


import os

from flask import Flask, render_template, request
from dotenv import load_dotenv

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

from src.helper import download_hugging_face_embeddings

from src.prompt import (
    contextualize_q_prompt,
    qa_prompt,
)


# ============================================================
# 1. Initialize Flask
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. Load Environment Variables
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)


if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing from the .env file."
    )


if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing from the .env file."
    )


# ============================================================
# 3. Load Embeddings
# ============================================================

embeddings = download_hugging_face_embeddings()


# ============================================================
# 4. Connect to Pinecone
# ============================================================

INDEX_NAME = "mediguide"


docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)


# ============================================================
# 5. Configure Retriever
# ============================================================

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
    },
)


# ============================================================
# 6. Configure GPT-4o
# ============================================================

chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ============================================================
# 7. Create History-Aware Retriever
# ============================================================

# This makes retrieval aware of the previous conversation.
#
# Example:
#
# What is acne?
# How can it be treated?
#
# becomes:
#
# How can acne be treated?

history_aware_retriever = create_history_aware_retriever(
    chat_model,
    retriever,
    contextualize_q_prompt,
)


# ============================================================
# 8. Create Question-Answering Chain
# ============================================================

question_answer_chain = create_stuff_documents_chain(
    chat_model,
    qa_prompt,
)


# ============================================================
# 9. Create Conversational RAG Chain
# ============================================================

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)


# ============================================================
# 10. Conversation Memory
# ============================================================

# Stores the current conversation while Flask is running.
#
# Example:
#
# HumanMessage("What is acne?")
# AIMessage("Acne is...")
# HumanMessage("How can it be treated?")
#
chat_history = []


# ============================================================
# 11. Home Route
# ============================================================

@app.route("/")
def home():

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

    global chat_history


    # Get user question.
    user_message = request.form.get(
        "msg",
        "",
    ).strip()


    if not user_message:

        return (
            "Please enter a valid healthcare question.",
            400,
        )


    try:

        print(
            f"\n[MediGuide] User: {user_message}"
        )


        # ----------------------------------------------------
        # Run Conversational RAG
        # ----------------------------------------------------

        response = rag_chain.invoke(
            {
                "input": user_message,
                "chat_history": chat_history,
            }
        )


        answer = response.get(
            "answer",
            "I could not generate a response.",
        )


        # ----------------------------------------------------
        # Save Question and Answer to Memory
        # ----------------------------------------------------

        chat_history.append(
            HumanMessage(
                content=user_message
            )
        )


        chat_history.append(
            AIMessage(
                content=answer
            )
        )


        # ----------------------------------------------------
        # Limit Conversation Memory
        # ----------------------------------------------------

        # Keep the most recent 12 messages.
        #
        # Approximately:
        # 6 user questions
        # +
        # 6 MediGuide answers

        if len(chat_history) > 12:

            chat_history = chat_history[-12:]


        print(
            f"[MediGuide] Assistant: {answer}"
        )


        print(
            f"[MediGuide] Memory size: "
            f"{len(chat_history)} messages\n"
        )


        return str(answer)


    except Exception as error:

        print(
            f"[MediGuide] Error: {error}"
        )


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

    global chat_history


    chat_history = []


    print(
        "[MediGuide] Conversation memory cleared."
    )


    return (
        "Conversation cleared.",
        200,
    )


# ============================================================
# 14. Health Check
# ============================================================

@app.route("/health")
def health():

    return {
        "application": "MediGuide",
        "status": "healthy",
        "model": "gpt-4o",
        "index": INDEX_NAME,
        "memory_messages": len(chat_history),
    }


# ============================================================
# 15. Run Flask
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=8080,
        debug=True,
    )