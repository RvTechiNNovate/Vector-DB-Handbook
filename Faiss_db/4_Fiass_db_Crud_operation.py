# Import necessary libraries
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings #,OpenAIEmbeddings  # Embedding models
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# Initialize the document loader
loader = TextLoader("path/to/your/textfile.txt")  # Load your text file
documents = loader.load()

# Initialize the text splitter to divide documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create a FAISS vector store from the documents
db = FAISS.from_documents(docs, embeddings)
print(f"Initial vector count: {db.index.ntotal}")

# ---- Create: Adding New Documents ----

# Prepare new documents to add
new_texts = ["New document text.", "Another document."]
new_docs = text_splitter.split_documents([Document(page_content=text) for text in new_texts])

# Add new documents to the FAISS vector store
db.add_documents(new_docs)
print(f"Vector count after adding: {db.index.ntotal}")

# ---- Read: Retrieving Data ----

# Perform a similarity search
query = "Your search query"
results = db.similarity_search(query)
print("Search results:")
for doc in results:
    print(f"Text: {doc.page_content}")

# Perform similarity search with score
results_with_scores = db.similarity_search_with_score(query)
print("Search results with scores:")
for doc, score in results_with_scores:
    print(f"Text: {doc.page_content}, Score: {score}")

# Search by embedding vector
query_vector = embeddings.embed_query(query)
results_by_vector = db.similarity_search_by_vector(query_vector)
print("Search results by embedding vector:")
for doc in results_by_vector:
    print(f"Text: {doc.page_content}")

# ---- Update: Modifying Data ----

# Remove old documents by ID (if you have IDs)
doc_ids_to_remove = ["id1", "id2"]
db.delete(doc_ids_to_remove)

# Add updated documents
updated_texts = ["Updated document text."]
updated_docs = text_splitter.split_documents([Document(page_content=text) for text in updated_texts])
db.add_documents(updated_docs)
print(f"Vector count after update: {db.index.ntotal}")

# ---- Delete: Removing Data ----

# Delete specific documents by ID
doc_ids_to_delete = ["id_to_delete"]
db.delete(doc_ids_to_delete)
print(f"Vector count after deletion: {db.index.ntotal}")

# Delete all documents (Reinitialize FAISS with an empty list of documents)
db = FAISS.from_documents([], embeddings)
print(f"Vector count after deleting all: {db.index.ntotal}")

# Optionally, save and load the FAISS index for persistence
db.save_local("faiss_index")  # Save the index
new_db = FAISS.load_local("faiss_index", embeddings)  # Load the index

# Verify the loaded index
docs_loaded = new_db.similarity_search(query)
print("Search results from loaded index:")
for doc in docs_loaded:
    print(f"Text: {doc.page_content}")
