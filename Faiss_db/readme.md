# FAISS Overview and Setup

**FAISS (Facebook AI Similarity Search)** is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. Additionally, it includes supporting code for evaluation and parameter tuning.

## Setup

To use the FAISS vector database with LangChain, you need to install the following packages:

```bash
pip install -U langchain-community faiss-cpu langchain-openai tiktoken
```

You can also install `faiss-gpu` if you want to use the GPU-enabled version.

### Setting Up OpenAI API Key

Since we're using OpenAI for embeddings, you'll need an OpenAI API Key:

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
```

## Document Ingestion

### Import Necessary Libraries

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```

### Load and Split Documents

```python
loader = TextLoader("../../modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
```

### Generate Embeddings and Initialize FAISS

```python
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)
```

## Querying the Vectorstore

### Similarity Search

```python
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

### Using FAISS as a Retriever

```python
retriever = db.as_retriever()
docs = retriever.invoke(query)
print(docs[0].page_content)
```

### Similarity Search with Score

FAISS also provides the `similarity_search_with_score` method, which returns the documents along with their similarity scores:

```python
docs_and_scores = db.similarity_search_with_score(query)
print(docs_and_scores[0])
```

### Search by Embedding Vector

```python
embedding_vector = embeddings.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)
```

## Saving and Loading the FAISS Index

You can save and load a FAISS index, which is useful for not having to recreate it every time.

### Save the Index

```python
db.save_local("faiss_index")
```

### Load the Index

```python
new_db = FAISS.load_local("faiss_index", embeddings)
docs = new_db.similarity_search(query)
print(docs[0])
```

## Serializing and De-Serializing to Bytes

You can pickle the FAISS Index by using the following functions:

```python
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

pkl = db.serialize_to_bytes()  # Serializes the FAISS index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl)  # Load the index
```

## Merging FAISS Vectorstores

You can merge two FAISS vectorstores:

```python
db1 = FAISS.from_texts(["foo"], embeddings)
db2 = FAISS.from_texts(["bar"], embeddings)

db1.merge_from(db2)
```

## Similarity Search with Filtering

FAISS vectorstore supports filtering by fetching more results than needed and then filtering them. Here’s an example:

```python
from langchain_core.documents import Document

list_of_documents = [
    Document(page_content="foo", metadata=dict(page=1)),
    Document(page_content="bar", metadata=dict(page=1)),
    Document(page_content="foo", metadata=dict(page=2)),
    Document(page_content="barbar", metadata=dict(page=2)),
    Document(page_content="foo", metadata=dict(page=3)),
    Document(page_content="bar burr", metadata=dict(page=3)),
    Document(page_content="foo", metadata=dict(page=4)),
    Document(page_content="bar bruh", metadata=dict(page=4)),
]
db = FAISS.from_documents(list_of_documents, embeddings)
results_with_scores = db.similarity_search_with_score("foo")
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
```

### Filtering Results

You can filter results based on metadata:

```python
results_with_scores = db.similarity_search_with_score("foo", filter=dict(page=1))
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
```

### Filtering with MMR

```python
results = db.max_marginal_relevance_search("foo", filter=dict(page=1))
for doc in results:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
```

## Deleting Records from Vectorstore

You can delete records from the vectorstore:

```python
print("count before:", db.index.ntotal)
db.delete([db.index_to_docstore_id[0]])
print("count after:", db.index.ntotal)
```

This markdown provides a comprehensive guide on setting up and using FAISS with LangChain, including querying, filtering, saving/loading, and managing your FAISS vectorstore.