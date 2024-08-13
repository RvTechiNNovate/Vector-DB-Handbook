Here's a detailed `README.md` file for your Qdrant setup and usage, including setup instructions, CRUD operations, and search functionalities:

```markdown
# Qdrant Vector Store Setup and Usage

## Overview

Qdrant is a high-performance vector search engine that allows you to perform similarity searches on large sets of vector embeddings. This guide provides instructions on how to set up Qdrant using various methods, perform CRUD operations, and execute different types of searches.

## Table of Contents

1. [Setup](#setup)
   - [Docker Setup](#docker-setup)
   - [Local In-Memory Setup](#local-in-memory-setup)
   - [Qdrant Cloud Setup](#qdrant-cloud-setup)
   - [On-Disk Storage](#on-disk-storage)
2. [CRUD Operations](#crud-operations)
   - [Add Documents](#add-documents)
   - [Delete Documents](#delete-documents)
3. [Search Types](#search-types)
   - [Dense Vector Search](#dense-vector-search)
   - [Sparse Vector Search](#sparse-vector-search)
   - [Hybrid Vector Search](#hybrid-vector-search)
   - [Metadata Filtering](#metadata-filtering)
   - [Search with Scores](#search-with-scores)
4. [Additional Resources](#additional-resources)

## Setup

### Docker Setup

To set up Qdrant using Docker:

1. **Pull the Qdrant Docker image:**

   ```sh
   docker pull qdrant/qdrant
   ```

2. **Run the Qdrant Docker container:**

   ```sh
   docker run -p 6333:6333 qdrant/qdrant
   ```

   This will start Qdrant on `http://localhost:6333`.

### Local In-Memory Setup

To set up Qdrant using an in-memory client:

```python
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")
```

### Qdrant Cloud Setup

To use Qdrant Cloud:

1. **Obtain your Qdrant Cloud URL and API key from the Qdrant Cloud dashboard.**

2. **Initialize Qdrant with your cloud URL and API key:**

   ```python
   from langchain_qdrant import QdrantVectorStore

   url = "<---qdrant cloud cluster url here --->"
   api_key = "<---api key here--->"

   vector_store = QdrantVectorStore.from_documents(
       docs,
       embedding=embeddings,
       url=url,
       prefer_grpc=True,
       api_key=api_key,
       collection_name="my_documents"
   )
   ```

### On-Disk Storage

To set up Qdrant with on-disk storage:

```python
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

client = QdrantClient(path="/tmp/langchain_qdrant")

client.create_collection(
    collection_name="demo_collection",
    vectors_config={"size": 3072, "distance": "Cosine"}
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings
)
```

## CRUD Operations

### Add Documents

To add documents to your vector store:

```python
from uuid import uuid4
from langchain_core.documents import Document

document_1 = Document(page_content="Sample content.", metadata={"source": "example"})

documents = [document_1, ...]  # Add your documents here
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
```

### Delete Documents

To delete documents from your vector store:

```python
vector_store.delete(ids=[uuids[-1]])
```

## Search Types

### Dense Vector Search

To perform a dense vector search:

```python
results = vector_store.similarity_search("query text", k=2)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

### Sparse Vector Search

To perform a sparse vector search:

1. **Install the FastEmbed package:**

   ```sh
   pip install fastembed
   ```

2. **Set up the sparse embeddings and perform the search:**

   ```python
   from langchain_qdrant import FastEmbedSparse, RetrievalMode

   sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")

   vector_store = QdrantVectorStore.from_documents(
       docs,
       sparse_embedding=sparse_embeddings,
       location=":memory:",
       collection_name="my_documents",
       retrieval_mode=RetrievalMode.SPARSE
   )

   query = "query text"
   results = vector_store.similarity_search(query)
   for res in results:
       print(f"* {res.page_content} [{res.metadata}]")
   ```

### Hybrid Vector Search

To perform a hybrid vector search:

```python
from langchain_qdrant import FastEmbedSparse, RetrievalMode

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")

vector_store = QdrantVectorStore.from_documents(
    docs,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    location=":memory:",
    collection_name="my_documents",
    retrieval_mode=RetrievalMode.HYBRID
)

query = "query text"
results = vector_store.similarity_search(query)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

### Metadata Filtering

To perform a search with metadata filtering:

```python
from qdrant_client.http import models

results = vector_store.similarity_search(
    query="query text",
    k=2,
    filter=models.Filter(
        should=[
            models.FieldCondition(
                key="metadata_key",
                match=models.MatchValue(value="desired_value")
            ),
        ]
    )
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

### Search with Scores

To perform a search and receive similarity scores:

```python
results = vector_store.similarity_search_with_score(
    query="query text",
    k=1
)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
```

## Additional Resources

- **[Qdrant Documentation](https://qdrant.tech/documentation/)**
- **[LangChain Documentation](https://langchain.com/docs/)**
- **[FastEmbed Documentation](https://fastembed.readthedocs.io/en/latest/)**
- **[Docker Hub: Qdrant](https://hub.docker.com/r/qdrant/qdrant)**

This guide covers the essentials of setting up and using Qdrant with various configurations and search methods. For further details, please refer to the documentation of Qdrant and related libraries.
```

### **Explanation**

1. **Setup**: Instructions for setting up Qdrant using Docker, local in-memory, Qdrant Cloud, and on-disk storage.

2. **CRUD Operations**: How to add and delete documents in the Qdrant vector store.

3. **Search Types**: Demonstrates different search types (dense, sparse, hybrid) and includes metadata filtering and searching with scores.

4. **Additional Resources**: Links to further documentation and resources.

Feel free to adapt or expand on these sections as needed!