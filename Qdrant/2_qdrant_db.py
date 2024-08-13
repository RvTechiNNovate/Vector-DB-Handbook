# qdrant_management.py

# Install necessary libraries
# Run these commands in your terminal: 
# pip install -qU langchain-huggingface qdrant-client fastembed

from uuid import uuid4
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Initialize Hugging Face embeddings
def initialize_embeddings():
    """
    Initialize Hugging Face embeddings using a pre-trained model.
    
    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings initialized with a specific model.
    """
    return HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

# Set up Qdrant with in-memory storage
def setup_in_memory_storage(embeddings):
    """
    Set up Qdrant with in-memory storage.
    
    Args:
        embeddings (HuggingFaceEmbeddings): The embeddings to use for vector representation.

    Returns:
        QdrantVectorStore: An instance of QdrantVectorStore with in-memory storage.
    """
    client = QdrantClient(":memory:")
    
    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    
    return QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=embeddings,
    )

# Add documents to the vector store
def add_documents(vector_store):
    """
    Add documents to the vector store.
    
    Args:
        vector_store (QdrantVectorStore): The vector store to which documents will be added.
    """
    documents = [
        Document(page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.", metadata={"source": "tweet"}),
        Document(page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.", metadata={"source": "news"}),
        Document(page_content="Building an exciting new project with LangChain - come check it out!", metadata={"source": "tweet"}),
        Document(page_content="Robbers broke into the city bank and stole $1 million in cash.", metadata={"source": "news"}),
        Document(page_content="Wow! That was an amazing movie. I can't wait to see it again.", metadata={"source": "tweet"}),
        Document(page_content="Is the new iPhone worth the price? Read this review to find out.", metadata={"source": "website"}),
        Document(page_content="The top 10 soccer players in the world right now.", metadata={"source": "website"}),
        Document(page_content="LangGraph is the best framework for building stateful, agentic applications!", metadata={"source": "tweet"}),
        Document(page_content="The stock market is down 500 points today due to fears of a recession.", metadata={"source": "news"}),
        Document(page_content="I have a bad feeling I am going to get deleted :(", metadata={"source": "tweet"}),
    ]
    
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    return uuids

# Delete documents from the vector store
def delete_documents(vector_store, ids):
    """
    Delete documents from the vector store by their IDs.
    
    Args:
        vector_store (QdrantVectorStore): The vector store from which documents will be deleted.
        ids (list): List of document IDs to be deleted.
    
    Returns:
        bool: True if documents were successfully deleted, False otherwise.
    """
    return vector_store.delete(ids=ids)

# Query the vector store
def query_vector_store(vector_store, query):
    """
    Perform a similarity search on the vector store.
    
    Args:
        vector_store (QdrantVectorStore): The vector store to query.
        query (str): The query string for the search.
    
    Returns:
        list: List of retrieved documents.
    """
    results = vector_store.similarity_search(query, k=2)
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")

# Use different search modes
def search_with_modes(vector_store, embeddings, query):
    """
    Perform searches using different retrieval modes.
    
    Args:
        vector_store (QdrantVectorStore): The vector store to search.
        embeddings (HuggingFaceEmbeddings): The embeddings to use for dense vector search.
        query (str): The query string for the search.
    """
    # Dense Vector Search
    dense_qdrant = QdrantVectorStore.from_documents(
        docs=[],
        embedding=embeddings,
        location=":memory:",
        collection_name="my_documents",
        retrieval_mode=RetrievalMode.DENSE,
    )
    print("Dense Search Results:")
    dense_qdrant.similarity_search(query)

    # Sparse Vector Search
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")
    sparse_qdrant = QdrantVectorStore.from_documents(
        docs=[],
        sparse_embedding=sparse_embeddings,
        location=":memory:",
        collection_name="my_documents",
        retrieval_mode=RetrievalMode.SPARSE,
    )
    print("Sparse Search Results:")
    sparse_qdrant.similarity_search(query)

    # Hybrid Vector Search
    hybrid_qdrant = QdrantVectorStore.from_documents(
        docs=[],
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        location=":memory:",
        collection_name="my_documents",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    print("Hybrid Search Results:")
    hybrid_qdrant.similarity_search(query)

def main():
    """
    Main function to demonstrate vector store management.
    """
    embeddings = initialize_embeddings()
    
    # Set up Qdrant with in-memory storage
    vector_store = setup_in_memory_storage(embeddings)
    
    # Add documents to the vector store
    uuids = add_documents(vector_store)
    
    # Delete the last document
    delete_documents(vector_store, ids=[uuids[-1]])
    
    # Perform a similarity search
    query_vector_store(vector_store, "LangChain provides abstractions to make working with LLMs easy")
    
    # Perform searches with different modes
    search_with_modes(vector_store, embeddings, "What did the president say about Ketanji Brown Jackson")

if __name__ == "__main__":
    main()
