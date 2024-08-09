# Import necessary modules
from langchain_community.document_loaders import TextLoader  # To load text files
from langchain_community.vectorstores import FAISS  # FAISS vector store for fast similarity search
from langchain_community.embeddings import HuggingFaceEmbeddings #,OpenAIEmbeddings  # Embedding models

from langchain_text_splitters import RecursiveCharacterTextSplitter  # To split documents into chunks

# Step 1: Load the text file
loader = TextLoader("/content/abc.txt")  # Replace with your file path
documents = loader.load()  # Load documents from the file

# Step 2: Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)  # Split with a max chunk size of 100 characters
docs = text_splitter.split_documents(documents)  # Split the loaded documents

# Step 3: Initialize the embedding model
# Uncomment this to use OpenAI embeddings
# embeddings = OpenAIEmbeddings()

# Use HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 4: Create a FAISS vector store from the documents
db = FAISS.from_documents(docs, embeddings)  # Store the document vectors in the FAISS index



db.save_local("faiss_index")

new_db = FAISS.load_local("faiss_index", embeddings)
query = "What did the president say about Ketanji Brown Jackson"
docs = new_db.similarity_search(query)