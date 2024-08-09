# Import necessary modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# Initialize the HuggingFace embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize the Chroma vector store
collection_name = 'example_collection'
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Directory to persist data
)

# Basic CRUD Operations

# 1. Create: Adding Data to the Vector Store
texts = ["Text 1", "Text 2", "Text 3"]
metadata_list = [{"source": "source1"}, {"source": "source2"}, {"source": "source3"}]
uuids = [str(uuid4()) for _ in range(len(texts))]

# Add texts to the vector store
vector_store.add_texts(texts=texts, ids=uuids, metadatas=metadata_list)

# 2. Read: Retrieving Data from the Vector Store Depends on your search types
results = vector_store.similarity_search(
    query="Example query",
    k=2,  # Number of results to return
    filter={"source": "source1"},  # Optional filter by metadata
)

# Display the results
for res in results:
    print(f"Text: {res['text']} - Metadata: {res['metadata']}")

# 3. Update: Modifying Data in the Vector Store
# Assuming you want to update a text with a specific ID
updated_text = "Updated Text 1"
updated_metadata = {"source": "updated_source1"}
vector_store.add_texts(texts=[updated_text], ids=[uuids[0]], metadatas=[updated_metadata])

# 4. Delete: Removing Data from the Vector Store
# Delete an entry based on its ID
vector_store.delete(ids=[uuids[1]])

# 5. Delete Collection: Removing an Entire Collection
# Delete the entire collection
vector_store.delete_collection()

# Note: This operation is irreversible, so use it with caution.
