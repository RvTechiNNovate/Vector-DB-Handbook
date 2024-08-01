FAISS (Facebook AI Similarity Search) is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors. It’s particularly well-suited for large-scale search applications where you need to handle high-dimensional vector data. Here’s a detailed guide on setting up and using FAISS:

### **1. Installation**

To get started with FAISS, you need to install the library. You can install it via pip:

```bash
pip install faiss-cpu  # For CPU version
pip install faiss-gpu  # For GPU version (if you have a compatible GPU)
```

### **2. Basic Usage**

Here’s a basic overview of how to use FAISS for indexing and searching vectors:

#### **a. Import FAISS and Create an Index**

```python
import faiss
import numpy as np

# Define the dimension of the vectors
d = 128  # Example dimension

# Create a FAISS index
index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean distance)
```

#### **b. Adding Vectors to the Index**

```python
# Generate some random vectors
num_vectors = 1000
vectors = np.random.random((num_vectors, d)).astype('float32')

# Add vectors to the index
index.add(vectors)
```

#### **c. Performing a Search**

```python
# Create a query vector
query_vector = np.random.random((1, d)).astype('float32')

# Perform a search (find the 5 nearest neighbors)
k = 5
distances, indices = index.search(query_vector, k)

# Print the results
print("Distances:", distances)
print("Indices:", indices)
```

### **3. Advanced Indexing**

FAISS supports several advanced indexing techniques to improve performance and scalability.

#### **a. Indexing with HNSW (Hierarchical Navigable Small World)**

HNSW is a popular method for approximate nearest neighbor search:

```python
# Create an HNSW index
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32 is the number of neighbors for HNSW

# Add vectors to the index
index_hnsw.add(vectors)

# Perform a search
distances, indices = index_hnsw.search(query_vector, k)

print("Distances:", distances)
print("Indices:", indices)
```

#### **b. Using IVF (Inverted File Index)**

IVF is useful for large-scale data:

```python
# Create an IVF index
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(d)  # Quantizer for IVF
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train the index (required before adding vectors)
index_ivf.train(vectors)

# Add vectors to the index
index_ivf.add(vectors)

# Perform a search
distances, indices = index_ivf.search(query_vector, k)

print("Distances:", distances)
print("Indices:", indices)
```

### **4. GPU Acceleration**

If you have a GPU and want to use it for FAISS, you can leverage GPU capabilities:

#### **a. GPU Setup**

```python
import faiss
import faiss.contrib.torch_utils

# Create a FAISS index
index_cpu = faiss.IndexFlatL2(d)

# Move the index to GPU
res = faiss.StandardGpuResources()  # Initialize GPU resources
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # Move index to GPU

# Add vectors and search as usual
index_gpu.add(vectors)
distances, indices = index_gpu.search(query_vector, k)

print("Distances:", distances)
print("Indices:", indices)
```

### **5. Saving and Loading Indexes**

You can save and load FAISS indexes for persistence:

#### **a. Save an Index**

```python
faiss.write_index(index, 'index_file.index')
```

#### **b. Load an Index**

```python
index = faiss.read_index('index_file.index')
```

### **6. Example: Full Workflow**

Here’s a complete example that covers the creation of an index, adding data, and querying:

```python
import faiss
import numpy as np

# Define dimensions and create an index
d = 128
index = faiss.IndexFlatL2(d)

# Generate and add random vectors
num_vectors = 1000
vectors = np.random.random((num_vectors, d)).astype('float32')
index.add(vectors)

# Create a query vector and search
query_vector = np.random.random((1, d)).astype('float32')
k = 5
distances, indices = index.search(query_vector, k)

# Print results
print("Distances:", distances)
print("Indices:", indices)
```

### **7. Resources**

- **FAISS Documentation**: [FAISS Documentation](https://faiss.ai/docs/)
- **FAISS GitHub**: [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)

FAISS is a powerful tool for similarity search and can be tailored to fit a wide range of use cases, from small-scale projects to large-scale deployments. If you have specific requirements or need further assistance, feel free to ask!