The `Chroma` vector store in the LangChain framework provides several methods for searching and retrieving data beyond just `similarity_search`. Here are some common methods:

### 1. **`similarity_search`**
   - This method searches for documents or texts that are similar to the given query, based on their vector embeddings.
   - You can use the `filter` parameter to narrow down the results based on metadata.

   ```python
   results = vector_store.similarity_search(
       query="LangChain provides abstractions to make working with LLMs easy",
       k=2,  # Number of similar documents to return
       filter={"source": "tweet"},
   )
   ```

### 2. **`similarity_search_with_score`**
   - This method works similarly to `similarity_search`, but it also returns a similarity score along with each result.
   - The score represents how close each result is to the query.

   ```python
   results = vector_store.similarity_search_with_score(
       query="LangChain provides abstractions to make working with LLMs easy",
       k=2,
       filter={"source": "tweet"},
   )
   for res, score in results:
       print(f"* {res['text']} [{res['metadata']}] with score {score}")
   ```

### 3. **`max_marginal_relevance_search` (MMR)**
   - This method is used to return results that maximize diversity while maintaining relevance to the query.
   - MMR is helpful when you want to avoid redundancy in the results by ensuring that they are not only relevant but also different from each other.

   ```python
   results = vector_store.max_marginal_relevance_search(
       query="LangChain provides abstractions to make working with LLMs easy",
       k=2,
       lambda_mult=0.5,  # Controls the trade-off between diversity and relevance
       filter={"source": "tweet"},
   )
   ```

### 4. **`semantic_search`**
   - Some implementations or configurations might support a `semantic_search` method, which is specifically optimized for deep semantic similarity. It’s very similar to `similarity_search` but may use more advanced techniques or models.

   ```python
   results = vector_store.semantic_search(
       query="LangChain provides abstractions to make working with LLMs easy",
       k=2,
       filter={"source": "tweet"},
   )
   ```

### 5. **`query_by_embedding`**
   - If you have the embedding vector directly, you can use it to perform a search without needing to pass a raw text query.
   - This method directly compares the provided embedding against the stored embeddings.

   ```python
   query_embedding = embeddings.embed_query("LangChain provides abstractions to make working with LLMs easy")
   results = vector_store.query_by_embedding(
       query_embedding=query_embedding,
       k=2,
       filter={"source": "tweet"},
   )
   ```

### 6. **`knn_search` (k-Nearest Neighbors)**
   - This method retrieves the top `k` nearest neighbors for a given query.
   - It might not be explicitly named as `knn_search` in all implementations but is conceptually similar to `similarity_search`.

   ```python
   results = vector_store.knn_search(
       query="LangChain provides abstractions to make working with LLMs easy",
       k=2,
   )
   ```

### Summary of Search Methods:
- **`similarity_search`**: General-purpose search for similar documents/texts.
- **`similarity_search_with_score`**: Returns results with similarity scores.
- **`max_marginal_relevance_search`**: Ensures diverse and relevant results.
- **`query_by_embedding`**: Search using a precomputed embedding vector.
- **`knn_search`**: Retrieve top `k` nearest neighbors for a query.

These methods allow you to tailor your search based on the specific needs of your application, whether you're optimizing for relevance, diversity, or using a precomputed embedding.