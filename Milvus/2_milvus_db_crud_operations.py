# %pip install -qU langchain-community langchain_milvus

# Import necessary libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from uuid import uuid4

# Step 1: Initialize the embedding model
# Using the 'sentence-transformers/all-mpnet-base-v2' model for generating embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 2: Set up the Milvus vector store
# Using Milvus Lite where everything is stored in a local file. If you have a Milvus server, use its URI.
URI = "./milvus_example.db"
collection_name = "example_collection"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name=collection_name,
)

# CRUD Operations

# CREATE (Adding Documents)

# Step 3: Create some example documents with metadata
documents = [
    Document(page_content="I worked at Kensho", metadata={"namespace": "harrison"}),
    Document(page_content="I worked at Facebook", metadata={"namespace": "ankush"}),
    Document(page_content="I worked at SDSD", metadata={"namespace": "ankush"}),
]

# Generate unique IDs for each document
uuids = [str(uuid4()) for _ in range(len(documents))]

# Step 4: Add documents to the vector store
vector_store.add_documents(documents=documents, ids=uuids)
print("Documents added successfully.")

# READ (Retrieving Documents)

# Step 5: Perform a similarity search
# Searching for documents similar to the query string
query = "Where did I work?"
results = vector_store.similarity_search(
    query=query,
    k=2,  # Number of similar documents to return
    filter={"namespace": "ankush"},  # Filter by 'namespace'
)

# Print the results
print("\nSimilarity Search Results:")
for res in results:
    print(f"Content: {res.page_content}, Metadata: {res.metadata}")

# UPDATE (Modifying Documents)

# Step 6: Update a document's content or metadata
# Example: Updating the content of the first document 

                        # WIP


# DELETE (Removing Documents)

# Step 8: Delete a document by ID
# Deleting the last document added
vector_store.delete(ids=[uuids[-1]])
print("\nDocument deleted successfully.")

# Step 9: Verify the deletion by attempting to retrieve the deleted document
remaining_results = vector_store.similarity_search(query="Where did I work?", k=3)
print("\nRemaining Documents After Deletion:")
for res in remaining_results:
    print(f"Content: {res.page_content}, Metadata: {res.metadata}")

# Additional Operations

# Step 10: Use a partitioned search with a retriever
# Here, we retrieve only documents where 'namespace' is "ankush"
retriever = vector_store.as_retriever(search_kwargs={"expr": 'namespace == "ankush"'})
partitioned_results = retriever.invoke(query)

# Print partitioned search results
print("\nPartitioned Search Results:")
for res in partitioned_results:
    print(f"Content: {res.page_content}, Metadata: {res.metadata}")

