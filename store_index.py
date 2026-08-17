# ============================================================
# MediGuide - Knowledge Base Indexing
# AI-Powered Analysis for Intelligent Healthcare Assistance
# ============================================================
#
# This script prepares the medical knowledge base for MediGuide.
#
# Main steps:
# 1. Load API keys from the .env file
# 2. Load medical PDF documents
# 3. Clean unnecessary document metadata
# 4. Split documents into smaller text chunks
# 5. Generate Hugging Face embeddings
# 6. Connect to Pinecone
# 7. Create the vector index if it does not already exist
# 8. Store medical document embeddings in Pinecone
#
# Authors:
# Saurabh Kumbhar - 25204974
# Azim Hassan   - 25203062
# ============================================================


# ----------------------------
# Required Imports
# ----------------------------

import os

from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)


# ============================================================
# 1. Load Environment Variables
# ============================================================

# Load API keys and other configuration values
# from the project's .env file.
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Validate that the required API keys are available.
if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing. "
        "Please add it to the .env file."
    )

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Please add it to the .env file."
    )


# ============================================================
# 2. Load Medical PDF Documents
# ============================================================

# Load all PDF files stored inside the data/ directory.
extracted_data = load_pdf_file(
    data="data/"
)

print(
    f"Loaded {len(extracted_data)} medical document pages."
)


# ============================================================
# 3. Clean Document Metadata
# ============================================================

# Keep only essential metadata such as the source file path.
# This makes the document objects cleaner before indexing.
filtered_data = filter_to_minimal_docs(
    extracted_data
)

print(
    f"Prepared {len(filtered_data)} cleaned documents."
)


# ============================================================
# 4. Split Documents into Text Chunks
# ============================================================

# Split the medical documents into smaller overlapping chunks.
# Smaller chunks improve semantic search and retrieval accuracy.
text_chunks = text_split(
    filtered_data
)

print(
    f"Created {len(text_chunks)} text chunks."
)


# ============================================================
# 5. Generate Hugging Face Embeddings
# ============================================================

# Load the sentence-transformer embedding model.
#
# MediGuide uses:
# sentence-transformers/all-MiniLM-L6-v2
#
# The model converts each text chunk into a
# 384-dimensional numerical vector.
embeddings = download_hugging_face_embeddings()

print("Hugging Face embedding model loaded successfully.")


# ============================================================
# 6. Initialize Pinecone
# ============================================================

# Create the Pinecone client using the API key
# stored securely in the .env file.
pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# ============================================================
# 7. Create the MediGuide Vector Index
# ============================================================

# Name of the Pinecone vector index.
#
# The index stores embeddings generated from the
# medical knowledge-base documents.
index_name = "mediguide"


# Create the index only if it does not already exist.
if not pc.has_index(index_name):

    pc.create_index(
        name=index_name,

        # all-MiniLM-L6-v2 generates
        # 384-dimensional embeddings.
        dimension=384,

        # Cosine similarity measures semantic similarity
        # between the user's question and document chunks.
        metric="cosine",

        # Configure Pinecone serverless infrastructure.
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    print(
        f"Pinecone index '{index_name}' created successfully."
    )

else:
    print(
        f"Pinecone index '{index_name}' already exists."
    )


# Connect to the Pinecone index.
index = pc.Index(index_name)


# ============================================================
# 8. Store Medical Document Embeddings in Pinecone
# ============================================================

# Convert all medical text chunks into embeddings
# and store them inside the Pinecone vector database.
#
# PineconeVectorStore connects:
#
# Medical Text
#      ↓
# Hugging Face Embeddings
#      ↓
# Vector Representation
#      ↓
# Pinecone Vector Database
#
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)


print(
    f"Successfully indexed {len(text_chunks)} medical "
    f"text chunks in Pinecone."
)

print("MediGuide knowledge base is ready.")


