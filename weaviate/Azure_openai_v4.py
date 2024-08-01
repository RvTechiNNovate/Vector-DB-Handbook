import weaviate
from weaviate.classes.config import Configure
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Weaviate client
client = weaviate.connect_to_local()

def create_schema(class_name, vectorizer_name, resource_name, deployment_id):
    """
    Creates a schema in Weaviate with a specified class name and vectorizer configuration.
    
    Args:
        class_name (str): The name of the class to be created in Weaviate.
        vectorizer_name (str): The name of the vectorizer.
        resource_name (str): The resource name for the Azure OpenAI vectorizer.
        deployment_id (str): The deployment ID for the Azure OpenAI model.
    """
    try:
        vectorizer_config = [
            Configure.NamedVectors.text2vec_azure_openai(
                name=vectorizer_name,
                source_properties=["text"],  # Change this to match your data schema
                resource_name=resource_name,
                deployment_id=deployment_id
            )
        ]

        client.collections.create(
            class_name,
            vectorizer_config=vectorizer_config
        )
        print(f"Schema '{class_name}' created successfully.")
    except Exception as e:
        print(f"Error creating schema '{class_name}': {e}")

def save_to_db(data, class_name):
    """
    Saves data to a Weaviate class.
    
    Args:
        data (list): List of dictionaries containing data to be stored.
        class_name (str): The name of the class to which data will be saved.
    """
    try:
        collection = client.collections.get(class_name)
        with collection.batch.dynamic() as batch:
            for i, d in enumerate(data):
                batch.add_object(properties=d)
                print(f"Added object {i}")
        print("Data has been stored successfully.")
    except Exception as e:
        print(f"Error saving data to class '{class_name}': {e}")

def delete_weaviate_class(class_name):
    """
    Deletes a Weaviate class.
    
    Args:
        class_name (str): The name of the class to be deleted.
    """
    try:
        client.collections.delete(class_name)
        print(f"Class '{class_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting class '{class_name}': {e}")

def load_and_process_documents(directory_path, chunk_size=1000, chunk_overlap=200):
    """
    Loads and processes documents from a directory.
    
    Args:
        directory_path (str): Path to the directory containing text files.
        chunk_size (int): Size of text chunks.
        chunk_overlap (int): Overlap between text chunks.
    
    Returns:
        list: Processed data ready to be saved to Weaviate.
    """
    loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(documents)
    return [{"source": doc.metadata["source"], "text": doc.page_content} for doc in split_documents]

def query_collection(class_name, query, limit=7):
    """
    Queries a Weaviate collection.
    
    Args:
        class_name (str): The name of the class to query.
        query (str): The query string to search for.
        limit (int): The maximum number of results to return.
    
    Returns:
        dict: The query results.
    """
    try:
        collection = client.collections.get(class_name)
        result = collection.query.hybrid(
            query=query,
            limit=limit
        )
        return result
    except Exception as e:
        print(f"Error querying class '{class_name}': {e}")
        return {}

def close_client():
    """
    Closes the Weaviate client connection.
    """
    client.close()
    print("Client connection closed.")

# Example Usage (not included in the function definitions)
if __name__ == "__main__":
    # Set azure OpenAI API key in docker compose file
    
    # Define parameters
    class_name = "DemoClass"
    resource_name = ""  # Replace with your Azure OpenAI resource name
    deployment_id = ""  # Replace with your Azure OpenAI deployment ID
    
    # Load and process documents
    data = load_and_process_documents('text_data')

    # Example function calls
    # Create schema
    create_schema(class_name, "azure_openai", resource_name, deployment_id)

    # Delete existing class
    delete_weaviate_class(class_name)

    # Save data
    save_to_db(data, class_name)

    # Perform a query
    query_result = query_collection(class_name, query="Sample query")
    print(query_result)

    # Close the client connection
    close_client()
