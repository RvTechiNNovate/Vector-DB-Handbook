## CRUD Operations

### 1. **Create**: Adding Data to the Vector Store

You can add data to the Chroma vector store using the `add_texts` method. This method allows you to add multiple texts along with their metadata and unique IDs.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# Initialize the HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize the Chroma vector store
vector_store = Chroma(
    collection_name='example_collection',
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Prepare texts, metadata, and unique IDs
texts = ["Text 1", "Text 2", "Text 3"]
metadata_list = [{"source": "source1"}, {"source": "source2"}, {"source": "source3"}]
uuids = [str(uuid4()) for _ in range(len(texts))]

# Add texts to the vector store
vector_store.add_texts(texts=texts, ids=uuids, metadatas=metadata_list)
```

### 2. **Read**: Retrieving Data from the Vector Store

You can retrieve data from the Chroma vector store using various search methods, such as `similarity_search`, `similarity_search_with_score`, and more.

```python
results = vector_store.similarity_search(
    query="Example query",
    k=2,  # Number of results to return
    filter={"source": "source1"},  # Optional filter
)

for res in results:
    print(f"Text: {res['text']} - Metadata: {res['metadata']}")
```

### 3. **Update**: Modifying Data in the Vector Store

Updating data involves re-adding the data with the same ID but with modified content or metadata. Currently, you might need to remove and then re-add the text if you want to update it.

```python
# Assuming you want to update a text with a specific ID
updated_text = "Updated Text 1"
updated_metadata = {"source": "updated_source1"}
vector_store.add_texts(texts=[updated_text], ids=["<existing-id>"], metadatas=[updated_metadata])
```

### 4. **Delete**: Removing Data from the Vector Store

To delete data from the vector store, you can use the `delete` method, which removes entries based on their IDs.

```python
vector_store.delete(ids=["<existing-id>"])
```

### 5. **Delete Collection**: Removing collection from the Vector Store
To delete an entire collection in the Chroma vector store, you can use the `delete_collection` method provided by the Chroma API. Here's how you can do it:

```python
# Initialize the Chroma vector store
vector_store = Chroma(
    collection_name='example_collection',
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Delete the entire collection
vector_store.delete_collection()
```

### Explanation:

- **Initialization**: You first initialize the `Chroma` vector store as you normally would, specifying the `collection_name` you want to work with.
- **Delete Collection**: By calling the `delete_collection` method, you can remove the entire collection, including all the data, metadata, and associated vectors.

This operation is irreversible, so make sure you really want to delete the entire collection before executing this command.



## Conclusion

This guide provides a basic overview of how to get started with the Chroma vector store, including how to perform CRUD operations. Chroma's intuitive API makes it a powerful tool for working with vector embeddings in AI-driven applications. 

