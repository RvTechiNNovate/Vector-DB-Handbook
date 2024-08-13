# qdrant_crud_operations.py

# Install necessary libraries
# Run these commands in your terminal:
# pip install -qU langchain-huggingface qdrant-client fastembed

from uuid import uuid4
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models

def initialize_embeddings():
    """
    Initialize Hugging Face embeddings using a pre-trained model.
    
    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings initialized with a specific model.
    """
    return HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

def setup_qdrant(embeddings):
    """
    Set up Qdrant with in-memory storage for demonstration.
    
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

def add_documents(vector_store):
    """
    Add documents to the vector store with unique IDs.
    
    Args:
        vector_store (QdrantVectorStore): The vector store to which documents will be added.
    
    Returns:
        list: List of document IDs.
    """
    documents = [
        Document(page_content="Chocolate chip pancakes and scrambled eggs.", metadata={"source": "tweet"}),
        Document(page_content="Tomorrow's weather: cloudy and overcast.", metadata={"source": "news"}),
        Document(page_content="Building an exciting project with LangChain!", metadata={"source": "tweet"}),
        Document(page_content="Robbers stole $1 million from the bank.", metadata={"source": "news"}),
        Document(page_content="Amazing movie, can't wait to see it again!", metadata={"source": "tweet"}),
    ]
    
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    return uuids

def read_documents(vector_store, query):
    """
    Query the vector store for similar documents.
    
    Args:
        vector_store (QdrantVectorStore): The vector store to query.
        query (str): The query string to search for.
    
    Returns:
        list: List of found documents.
    """
    results = vector_store.similarity_search(query=query, k=3)
    return results

def update_document(vector_store, old_id, new_document):
    """
    Update a document in the vector store by ID.
    
    Args:
        vector_store (QdrantVectorStore): The vector store containing the document to update.
        old_id (str): The ID of the document to update.
        new_document (Document): The new document to replace the old one.
    
    Returns:
        bool: True if the document was updated, False otherwise.
    """
    # Delete the old document
    vector_store.delete(ids=[old_id])
    
    # Add the new document
    result = vector_store.add_documents(documents=[new_document], ids=[old_id])
    return result

def delete_document(vector_store, document_id):
    """
    Delete a document from the vector store by ID.
    
    Args:
        vector_store (QdrantVectorStore): The vector store from which the document will be deleted.
        document_id (str): The ID of the document to delete.
    
    Returns:
        bool: True if the document was deleted, False otherwise.
    """
    return vector_store.delete(ids=[document_id])

def main():
    """
    Main function to demonstrate CRUD operations with Qdrant.
    """
    # Initialize embeddings
    embeddings = initialize_embeddings()
    
    # Set up Qdrant vector store
    vector_store = setup_qdrant(embeddings)
    
    # Add documents and retrieve their IDs
    document_ids = add_documents(vector_store)
    print(f"Documents added with IDs: {document_ids}")
    
    # Read (query) documents
    query_result = read_documents(vector_store, "LangChain project")
    print("Query Results:")
    for doc in query_result:
        print(f"* {doc.page_content} [{doc.metadata}]")
    
    # Update a document
    new_document = Document(page_content="New content for updated document.", metadata={"source": "updated"})
    if update_document(vector_store, document_ids[0], new_document):
        print(f"Document with ID {document_ids[0]} updated.")
    
    # Delete a document
    if delete_document(vector_store, document_ids[1]):
        print(f"Document with ID {document_ids[1]} deleted.")
    
    # Query again to see updates
    updated_query_result = read_documents(vector_store, "LangChain project")
    print("Updated Query Results:")
    for doc in updated_query_result:
        print(f"* {doc.page_content} [{doc.metadata}]")

if __name__ == "__main__":
    main()
