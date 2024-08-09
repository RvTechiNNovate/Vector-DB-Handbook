This guide provides a comprehensive overview of how to use ChromaDB effectively for managing and querying vector embeddings. Here's a structured approach to get started:

### **1. Installation**

Install ChromaDB using pip:

```bash
pip install chromadb
```

### **2. Setting Up ChromaDB**

Initialize the ChromaDB client to start interacting with the database:

```python
import chromadb

# Import necessary libraries and modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4

# Step 1: Initialize  embeddings
# Here, we're using the 'sentence-transformers/all-mpnet-base-v2' model for generating embeddings. You can use Open ai also.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# OR

# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



# Step 2: Initialize the Chroma vector store with a persistent directory
# `collection_name` specifies the collection we're working with.
# `embedding_function` is the embedding model used to generate embeddings.
# `persist_directory` is where Chroma will store its data on the local filesystem. Remove this parameter if persistence is not required.
collection_name = 'xyz'
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db1",  # Data will be saved in this directory
)


# Step 3: Create some example documents with metadata
# Each document contains text (page_content) and metadata (like source).
# The `id` field is set for each document for easy reference.
document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

# Step 4: Prepare documents and unique IDs for adding to the vector store
# Generate UUIDs for each document as unique identifiers.
documents = [document_1, document_2, document_3]
uuids = [str(uuid4()) for _ in range(len(documents))]

# Step 5: Add documents to the vector store
# The documents and their corresponding UUIDs are added to the Chroma vector store.
vector_store.add_documents(documents=documents, ids=uuids)

# Step 6: Perform a similarity search
# Here, we search for documents similar to the query string.
# The `filter` parameter is used to only search within documents that have the `source` set to "tweet".
results = vector_store.similarity_search(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,  # Number of similar documents to return
    filter={"source": "tweet"},  #[Optional] Filter documents by metadata 
)

# Step 7: Print the results
# The results from the similarity search are printed, showing the document content and metadata.
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")


############################################ For Storing Text Direct ###########################################


# Step 3: Prepare the data (texts and metadata)
# Texts to be added to the vector store
texts = [
    "I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    "The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    "Building an exciting new project with LangChain - come check it out!"
]

# Associated metadata for each text
metadata_list = [
    {"source": "tweet"},
    {"source": "news"},
    {"source": "tweet"}
]

# Generate unique IDs for each text
uuids = [str(uuid4()) for _ in range(len(texts))]

# Step 4: Add texts to the vector store with their corresponding metadata and IDs
vector_store.add_texts(texts=texts, ids=uuids, metadatas=metadata_list)

# Step 5: Perform a similarity search
# Searching for texts similar to the query string, filtering for texts from the "tweet" source.
results = vector_store.similarity_search(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,  # Number of similar texts to return
    filter={"source": "tweet"},  # Filter texts by metadata
)
```

- **API Keys**: Secure your API keys if using a hosted version.
- **Data Privacy**: Handle sensitive data with care and in compliance with regulations.
- **Efficiency**: Use vector normalization and ensure consistent dimensions for better performance.

This guide covers the essential aspects of setting up and using ChromaDB. Adjust the example code based on your specific needs and data. If you need more detailed instructions or run into any issues, feel free to ask!
