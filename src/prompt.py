# ============================================================
# MediGuide - Prompt Configuration
# AI-Powered Analysis for Intelligent Healthcare Assistance
# ============================================================
#
# This file contains:
# 1. Follow-up question reformulation prompt
# 2. Medical question-answering prompt
# 3. Conversation history placeholders
#
# Authors:
# Saurabh Kumbhar - 25204974
# Azim Hassan   - 25203062
# ============================================================


from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


# ============================================================
# 1. Follow-Up Question Reformulation Prompt
# ============================================================

# This prompt allows MediGuide to understand questions such as:
#
# User:
# "What is acne?"
#
# User:
# "How can it be treated?"
#
# The model internally reformulates the second question into:
#
# "How can acne be treated?"
#
# This improved standalone query is then sent to Pinecone.

contextualize_q_system_prompt = (
    "Given the conversation history and the latest user question, "
    "rewrite the latest question as a standalone question that can "
    "be understood without the previous conversation. "
    "Do not answer the question. "
    "Only reformulate the question when necessary."
)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            contextualize_q_system_prompt,
        ),

        # Insert previous conversation messages here.
        MessagesPlaceholder("chat_history"),

        (
            "human",
            "{input}",
        ),
    ]
)


# ============================================================
# 2. MediGuide Medical Question-Answering Prompt
# ============================================================

system_prompt = (
    "You are MediGuide, a medical information assistant. "
    "Answer the user's question using the retrieved medical context. "
    "Use conversation history when necessary to understand follow-up "
    "questions and references to previously discussed conditions. "
    "If the answer cannot be determined from the retrieved context, "
    "say that you do not know. "
    "Keep the response clear, concise, and medically grounded. "
    "Do not invent information that is not supported by the context."
    "\n\n"
    "Retrieved medical context:\n"
    "{context}"
)


# ============================================================
# 3. Question-Answering Prompt With Memory
# ============================================================

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),

        # Previous user and assistant messages.
        MessagesPlaceholder("chat_history"),

        (
            "human",
            "{input}",
        ),
    ]
)