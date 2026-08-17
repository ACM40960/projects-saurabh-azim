# ============================================================
# MediGuide - Helper Functions
# AI-Powered Analysis for Intelligent Healthcare Assistance
# ============================================================
#
# This module contains reusable helper functions for:
# 1. Loading medical PDF documents
# 2. Cleaning document metadata
# 3. Splitting documents into smaller text chunks
# 4. Creating Hugging Face embeddings
#
# Authors:
# Saurabh Kumbhar - 25204974
# Azim Hassan   - 25203062
# ============================================================


# ----------------------------
# Required Imports
# ----------------------------

from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document


# ============================================================
# 1. Load Medical PDF Documents
# ============================================================

def load_pdf_file(data: str) -> List[Document]:
    """
    Load all PDF files from the specified directory.

    Parameters
    ----------
    data : str
        Path to the directory containing medical PDF files.

    Returns
    -------
    List[Document]
        A list of LangChain Document objects extracted
        from the PDF files.
    """

    # DirectoryLoader searches the specified folder
    # and loads every file matching the *.pdf pattern.
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    # Extract text and metadata from all PDF documents.
    documents = loader.load()

    return documents


# ============================================================
# 2. Filter Document Metadata
# ============================================================

def filter_to_minimal_docs(
    docs: List[Document],
) -> List[Document]:
    """
    Reduce document metadata while preserving the original text.

    Only the source file path is retained in the metadata.
    This keeps the documents lightweight while maintaining
    traceability to the original medical source.

    Parameters
    ----------
    docs : List[Document]
        Documents extracted from the PDF files.

    Returns
    -------
    List[Document]
        Documents containing the original page content
        and only the source metadata.
    """

    minimal_docs: List[Document] = []

    for doc in docs:

        # Retrieve the original PDF source.
        source = doc.metadata.get("source", "unknown")

        # Create a simplified Document object.
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={
                "source": source
            },
        )

        minimal_docs.append(minimal_doc)

    return minimal_docs


# ============================================================
# 3. Split Documents into Text Chunks
# ============================================================

def text_split(
    extracted_data: List[Document],
) -> List[Document]:
    """
    Split medical documents into smaller overlapping text chunks.

    Smaller chunks improve semantic retrieval because the vector
    database can search individual sections instead of entire
    PDF pages or documents.

    Parameters
    ----------
    extracted_data : List[Document]
        Cleaned medical documents.

    Returns
    -------
    List[Document]
        Smaller document chunks ready for embedding.
    """

    # Configure the recursive text splitter.
    #
    # chunk_size:
    # Maximum size of each text chunk.
    #
    # chunk_overlap:
    # Amount of overlapping text between consecutive chunks.
    # Overlap helps preserve context between neighboring chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )

    # Split the documents into smaller sections.
    text_chunks = text_splitter.split_documents(
        extracted_data
    )

    return text_chunks


# ============================================================
# 4. Create Hugging Face Embeddings
# ============================================================

def download_hugging_face_embeddings():
    """
    Load the Hugging Face sentence-transformer embedding model.

    The 'all-MiniLM-L6-v2' model converts text into
    384-dimensional numerical vectors.

    These vectors are later stored in Pinecone and used
    for semantic similarity search.

    Returns
    -------
    HuggingFaceEmbeddings
        Configured LangChain embedding model.
    """

    # Load the pretrained Sentence Transformer model.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings