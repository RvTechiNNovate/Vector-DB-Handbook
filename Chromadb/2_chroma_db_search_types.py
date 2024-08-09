# Import necessary modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Initialize the HuggingFace embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize the Chroma vector store
collection_name = 'example_collection'
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# 1. Similarity Search
# Search for documents similar to the query based on vector embeddings
results = vector_store.similarity_search(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,  # Number of similar documents to return
    filter={"source": "tweet"},  # Optional filter by metadata
)
for res in results:
    print(f"Similarity Search Result: {res['text']} - Metadata: {res['metadata']}")

# 2. Similarity Search with Score
# Search for similar documents and return the similarity score along with the result
results_with_score = vector_store.similarity_search_with_score(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,  # Number of similar documents to return
    filter={"source": "tweet"},  # Optional filter by metadata
)
for res, score in results_with_score:
    print(f"Similarity Search with Score Result: {res['text']} - Metadata: {res['metadata']} - Score: {score}")

# 3. Max Marginal Relevance Search (MMR)
# Return results that maximize diversity while maintaining relevance to the query
results_mmr = vector_store.max_marginal_relevance_search(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,  # Number of similar documents to return
    lambda_mult=0.5,  # Controls the trade-off between diversity and relevance
    filter={"source": "tweet"},  # Optional filter by metadata
)
for res in results_mmr:
    print(f"MMR Search Result: {res['text']} - Metadata: {res['metadata']}")

# 4. Query by Embedding
# Search using a precomputed embedding vector instead of a raw text query
query_embedding = embeddings.embed_query("LangChain provides abstractions to make working with LLMs easy")
results_by_embedding = vector_store.query_by_embedding(
    query_embedding=query_embedding,
    k=2,  # Number of similar documents to return
    filter={"source": "tweet"},  # Optional filter by metadata
)
for res in results_by_embedding:
    print(f"Query by Embedding Result: {res['text']} - Metadata: {res['metadata']}")

# 5. k-Nearest Neighbors Search (kNN)
# Retrieve the top 'k' nearest neighbors for a given query
# This method might not be explicitly named 'knn_search' but is conceptually similar to 'similarity_search'
results_knn = vector_store.knn_search(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,  # Number of similar documents to return
)
for res in results_knn:
    print(f"kNN Search Result: {res['text']} - Metadata: {res['metadata']}")
