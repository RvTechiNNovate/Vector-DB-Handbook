# Import necessary libraries and modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4

# Step 1: Initialize  embeddings
# Here, we're using the 'sentence-transformers/all-mpnet-base-v2' model for generating embeddings. You can use Open ai also.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


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

# Here ROLE BASE ACCESS COTROL 
# Step 3: Create some documents with username which user can access it. pass it to its Metadata as shown below.
# Each document contains metadata with their username 
document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet",'user':'mayo'},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news",'user':'rv'},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet",'user':'rv'},
    id=3,
)

# Step 4: Prepare documents and unique IDs for adding to the vector store
# Generate UUIDs for each document as unique identifiers.
documents = [document_1, document_2, document_3]
uuids = [str(uuid4()) for _ in range(len(documents))]

# Step 5: Add documents to the vector store
# The documents and their corresponding UUIDs are added to the Chroma vector store.
vector_store.add_documents(documents=documents, ids=uuids)


# Step 6: Perform a similarity search for user 
# Here, we search for documents similar to the query string.
# The `filter` parameter is used to only search within documents that have a particular user.
results = vector_store.similarity_search(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,  # Number of similar documents to return
    filter={"user": "rv"},  #[Optional] Filter documents by usename
)

# Step 7: Print the results
# The results from the similarity search of the user 'rv' are printed, showing the document content and metadata.
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")


