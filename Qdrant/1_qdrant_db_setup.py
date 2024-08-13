# qdrant_setup.py

# Install necessary libraries
# Run this in your terminal: pip install -qU langchain-huggingface qdrant-client

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def initialize_embeddings():
    """Initialize Hugging Face embeddings."""
    return HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

def setup_in_memory_storage(embeddings):
    """Set up Qdrant with in-memory storage."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=embeddings,
    )
    return vector_store

def setup_on_disk_storage(embeddings):
    """Set up Qdrant with on-disk storage."""
    client = QdrantClient(path="/tmp/langchain_qdrant")
    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=embeddings,
    )
    return vector_store

def setup_on_premise_server(embeddings, url):
    """Set up Qdrant with an on-premise server deployment."""
    docs = []  # Add your documents here
    qdrant = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=url,
        prefer_grpc=True,  # Optionally use gRPC for performance
        collection_name="my_documents",
    )
    return qdrant

def setup_qdrant_cloud(embeddings, url, api_key):
    """Set up Qdrant with Qdrant Cloud."""
    docs = []  # Add your documents here
    qdrant = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=url,
        prefer_grpc=True,  # Optionally use gRPC for performance
        api_key=api_key,  # API key for accessing the cloud deployment
        collection_name="my_documents",
    )
    return qdrant

def setup_existing_collection(embeddings, url):
    """Connect to an existing Qdrant collection."""
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="my_documents",
        url=url,  # URL for your local Qdrant instance
    )
    return qdrant

def main():
    embeddings = initialize_embeddings()
    
    # Example setups
    print("Setting up in-memory storage...")
    in_memory_store = setup_in_memory_storage(embeddings)
    
    print("Setting up on-disk storage...")
    on_disk_store = setup_on_disk_storage(embeddings)
    
    # Replace with your Qdrant server URL
    url = "<---qdrant url here--->"
    print("Setting up on-premise server...")
    on_premise_store = setup_on_premise_server(embeddings, url)
    
    # Replace with your Qdrant Cloud URL and API key
    url = "<---qdrant cloud cluster url here--->"
    api_key = "<---api key here--->"
    print("Setting up Qdrant Cloud...")
    cloud_store = setup_qdrant_cloud(embeddings, url, api_key)
    
    # Replace with your local Qdrant instance URL
    url = "http://localhost:6333"
    print("Setting up existing collection...")
    existing_collection_store = setup_existing_collection(embeddings, url)
    
if __name__ == "__main__":
    main()
