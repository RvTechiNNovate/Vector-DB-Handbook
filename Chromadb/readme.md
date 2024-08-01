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

# Initialize a ChromaDB client
db = chromadb.Client()
```

### **3. Creating a Collection**

Collections in ChromaDB group vectors and their metadata. Create a collection with a name:

```python
# Create a new collection
collection_name = "my_collection"
collection = db.create_collection(name=collection_name)
```

### **4. Inserting Data**

Add vectors along with optional metadata into the collection:

```python
# Example vectors and metadata
vectors = [
    {"id": "vec1", "vector": [0.1, 0.2, 0.3], "metadata": {"label": "A"}},
    {"id": "vec2", "vector": [0.4, 0.5, 0.6], "metadata": {"label": "B"}}
]

# Insert data into the collection
for item in vectors:
    collection.add_vector(id=item["id"], vector=item["vector"], metadata=item["metadata"])
```

### **5. Querying Data**

Perform nearest neighbor searches to retrieve vectors similar to a query vector:

```python
query_vector = [0.3, 0.4, 0.5]

# Perform a nearest neighbor search
results = collection.search_vector(query_vector, top_k=1)

# Print the results
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}")
```

### **6. Updating Data**

Update the metadata of an existing vector:

```python
# Update vector with new metadata
collection.update_vector(id="vec1", metadata={"label": "Updated_A"})
```

### **7. Deleting Data**

Remove vectors from the collection:

```python
# Delete a vector by ID
collection.delete_vector(id="vec1")
```

### **8. Managing Collections**

List all collections and delete a collection if needed:

**List all collections:**

```python
collections = db.list_collections()
for collection in collections:
    print(collection.name)
```

**Delete a collection:**

```python
db.delete_collection(name="my_collection")
```

### **9. Handling Vector Embeddings**

Generate embeddings using a machine learning model. Here's an example using BERT:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Encode text to vectors
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Example usage
text = "Hello, world!"
embedding = get_embedding(text)
```

### **10. Example: Full Workflow**

Integrate ChromaDB with a model to encode and store vectors:

```python
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize ChromaDB client
db = chromadb.Client()

# Create a collection
collection = db.create_collection(name="text_embeddings")

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Encode text to vectors
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Add vectors to collection
texts = ["Hello, world!", "ChromaDB is cool!", "How are you?"]
for text in texts:
    vector = get_embedding(text)
    collection.add_vector(id=text, vector=vector, metadata={"text": text})

# Query the collection
query_text = "Hello!"
query_vector = get_embedding(query_text)
results = collection.search_vector(query_vector, top_k=2)

# Print results
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}, Metadata: {result['metadata']}")
```

### **11. Security and Best Practices**

- **API Keys**: Secure your API keys if using a hosted version.
- **Data Privacy**: Handle sensitive data with care and in compliance with regulations.
- **Efficiency**: Use vector normalization and ensure consistent dimensions for better performance.

This guide covers the essential aspects of setting up and using ChromaDB. Adjust the example code based on your specific needs and data. If you need more detailed instructions or run into any issues, feel free to ask!