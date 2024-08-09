# Import necessary libraries
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings #,OpenAIEmbeddings  # Embedding models

# Assuming 'db' is an already initialized FAISS vector store
# Initialize the embeddings model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# db = FAISS.from_documents(docs, embeddings)

# Define a query
query = "What did the president say about Ketanji Brown Jackson"

# ---- Similarity Search ----

# Perform a similarity search
docs = db.similarity_search(query)
print("Similarity search results:")
for doc in docs:
    print(f"Text: {doc.page_content}")

# ---- Similarity Search with Scores ----

# Perform a similarity search with scores
docs_and_scores = db.similarity_search_with_score(query)
print("Similarity search results with scores:")
for doc, score in docs_and_scores:
    print(f"Text: {doc.page_content}, Score: {score}")

# Example for another query to show results with scores
results_with_scores = db.similarity_search_with_score("foo")
print("Search results for 'foo' with scores:")
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
