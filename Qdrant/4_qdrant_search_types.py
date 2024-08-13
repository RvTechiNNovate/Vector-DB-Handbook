# qdrant_search_demo.py

# Install necessary libraries
# Run these commands in your terminal:
# pip install -qU langchain-huggingface qdrant-client fastembed

from uuid import uuid4
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Step 1: Initialize embeddings
def initialize_embeddings():
    return HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

# Step 2: Set up Qdrant Vector Store
def setup_qdrant(embeddings, sparse_embeddings=None, retrieval_mode=RetrievalMode.DENSE):
    client = QdrantClient(":memory:")  # Use in-memory storage for demonstration
    client.create_collection(
        collection_name="demo_collection",
        vectors_config={"size": 3072, "distance": "Cosine"}
    )
    return QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=retrieval_mode
    )

# Step 3: Add documents
def add_documents(vector_store):
    documents = [
        Document(page_content="Chocolate chip pancakes and scrambled eggs.", metadata={"source": "tweet"}),
        Document(page_content="Tomorrow's weather: cloudy and overcast.", metadata={"source": "news"}),
        Document(page_content="Building an exciting project with LangChain!", metadata={"source": "tweet"}),
        Document(page_content="Robbers stole $1 million from the bank.", metadata={"source": "news"}),
        Document(page_content="Amazing movie, can't wait to see it again!", metadata={"source": "tweet"})
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

# Step 4: Perform Dense Vector Search
def dense_vector_search(vector_store, query):
    print("\nDense Vector Search Results:")
    results = vector_store.similarity_search(query=query, k=2)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")

# Step 5: Perform Sparse Vector Search
def sparse_vector_search(vector_store, query):
    print("\nSparse Vector Search Results:")
    results = vector_store.similarity_search(query=query, k=2)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")

# Step 6: Perform Hybrid Vector Search
def hybrid_vector_search(vector_store, query):
    print("\nHybrid Vector Search Results:")
    results = vector_store.similarity_search(query=query, k=2)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")

# Step 7: Metadata Filtering
def metadata_filtering(vector_store, query, metadata_key, metadata_value):
    print("\nMetadata Filtering Results:")
    filter_condition = models.Filter(
        should=[
            models.FieldCondition(
                key=metadata_key,
                match=models.MatchValue(value=metadata_value)
            ),
        ]
    )
    results = vector_store.similarity_search(
        query=query,
        k=2,
        filter=filter_condition
    )
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")

# Step 8: Search with Scores
def search_with_score(vector_store, query, k=1):
    print("\nSearch with Score Results:")
    results = vector_store.similarity_search_with_score(query=query, k=k)
    for doc, score in results:
        print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

def main():
    # Initialize embeddings
    embeddings = initialize_embeddings()
    
    # Dense Vector Search
    dense_vector_store = setup_qdrant(embeddings, retrieval_mode=RetrievalMode.DENSE)
    add_documents(dense_vector_store)
    dense_vector_search(dense_vector_store, "LangChain project")
    
    # Sparse Vector Search
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")
    sparse_vector_store = setup_qdrant(sparse_embeddings=sparse_embeddings, retrieval_mode=RetrievalMode.SPARSE)
    add_documents(sparse_vector_store)
    sparse_vector_search(sparse_vector_store, "LangChain project")
    
    # Hybrid Vector Search
    hybrid_vector_store = setup_qdrant(embeddings, sparse_embeddings=sparse_embeddings, retrieval_mode=RetrievalMode.HYBRID)
    add_documents(hybrid_vector_store)
    hybrid_vector_search(hybrid_vector_store, "LangChain project")
    
    # Metadata Filtering
    metadata_filtered_store = setup_qdrant(embeddings, retrieval_mode=RetrievalMode.DENSE)
    add_documents(metadata_filtered_store)
    metadata_filtering(metadata_filtered_store, "best soccer players", "page_content", "The top 10 soccer players in the world right now.")
    
    # Search with Score
    scored_store = setup_qdrant(embeddings, retrieval_mode=RetrievalMode.DENSE)
    add_documents(scored_store)
    search_with_score(scored_store, "Will it be hot tomorrow", k=1)

if __name__ == "__main__":
    main()
