# %pip install -qU langchain-community langchain_milvus

# Import necessary libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document

# Step 1: Initialize the embedding model
# We're using the 'sentence-transformers/all-mpnet-base-v2' model for generating embeddings.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 2: Set up the Milvus vector store
# Using Milvus Lite where everything is stored in a local file. If you have a Milvus server, you can use its URI.
URI = "./milvus_example.db"

# Step 3: Create documents with content and metadata
# The 'namespace' field in metadata will be used as the partition key.
docs = [
    Document(page_content="I worked at Kensho", metadata={"namespace": "harrison"}),
    Document(page_content="I worked at Facebook", metadata={"namespace": "ankush"}),
    Document(page_content="I worked at SDSD", metadata={"namespace": "ankush"}),
]

# Step 4: Initialize the vector store with partitioning
# The 'namespace' field from the document metadata is used as the partition key.
vectorstore = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"uri": URI},
    drop_old=True,  # Drop old data if it exists in the same collection
    partition_key_field="namespace",  # Use the 'namespace' field for partitioning
)

# Step 5: Retrieve documents from a specific partition
# Here, we are retrieving only documents where 'namespace' is "ankush".
retriever = vectorstore.as_retriever(search_kwargs={"expr": 'namespace == "ankush"'})
results = retriever.invoke("Where did I work?")

# Print the retrieved documents
for res in results:
    print(f"Content: {res.page_content}, Metadata: {res.metadata}")
