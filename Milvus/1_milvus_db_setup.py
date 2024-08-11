# Install the required packages before using
# %pip install -qU langchain-community langchain_milvus

# Import necessary libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from uuid import uuid4
from langchain_core.documents import Document

# Step 1: Initialize the embedding model
# Here, we're using the 'sentence-transformers/all-mpnet-base-v2' model for generating embeddings.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 2: Set up the Milvus vector store
# If you have a Milvus server, use its URI, otherwise, Milvus Lite will store everything in a local file.
URI = "./milvus_example.db"
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
)

# Step 3: Create documents with content and metadata
document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

# Step 4: Generate UUIDs for each document
documents = [document_1, document_2, document_3]
uuids = [str(uuid4()) for _ in range(len(documents))]

# Step 5: Add the documents to the Milvus vector store
vector_store.add_documents(documents=documents, ids=uuids)

# Step 6 Perform a similarity search on the stored documents
# Filter the search to only return results with the "tweet" source in metadata
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)

# Print the results of the similarity search
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# Step 7: Use the vector store as a retriever with MMR search type
# MMR (Maximal Marginal Relevance) is used to reduce redundancy in the results.
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
