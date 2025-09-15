# Query the Chroma vector database and retrieve relevant content
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")

def load_vector_database():
    """Load the existing Chroma vector database."""
    print("Loading vector database...")
    
    if not os.path.exists("./chroma_db"):
        print("❌ Vector database not found. Please run create_database.py first.")
        return None
    
    try:
        # Initialize embeddings (same as used during creation)
        embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-base-en-v1.5",
            huggingfacehub_api_token=hf_api_key
        )
        
        # Load existing vector store
        vectorstore = Chroma(
            collection_name="pdf_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        print("✅ Vector database loaded successfully!")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error loading vector database: {str(e)}")
        return None

def retrieve_relevant_content(vectorstore, query, num_chunks=5):
    """Retrieve relevant content chunks based on the query."""
    print(f"Retrieving relevant content for: '{query}'")
    
    try:
        # Create retriever from vectorstore
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_chunks}
        )
        
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(query)
        
        if not relevant_docs:
            print("❌ No relevant content found.")
            return None, []
        
        # Combine content from all retrieved chunks
        combined_content = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        print(f"✅ Retrieved {len(relevant_docs)} relevant chunks")
        return combined_content, relevant_docs
        
    except Exception as e:
        print(f"❌ Error retrieving content: {str(e)}")
        return None, []

def search_database(query, num_chunks=5):
    """Main function to search the database and return relevant content."""
    # Load vector database
    vectorstore = load_vector_database()
    if not vectorstore:
        return None, []
    
    # Retrieve content
    content, docs = retrieve_relevant_content(vectorstore, query, num_chunks)
    return content, docs

def display_retrieved_content(content, docs):
    """Display the retrieved content in a formatted way."""
    print("\n" + "="*60)
    print("📖 RETRIEVED CONTENT")
    print("="*60)
    
    if not content:
        print("No content retrieved.")
        return
    
    print(f"\n📊 Found {len(docs)} relevant chunks:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', 'Unknown page')
        print(f"   {i}. {source} (Page {page})")
    
    print(f"\n📝 Combined Content ({len(content)} characters):")
    print("-" * 40)
    print(content[:1000] + "..." if len(content) > 1000 else content)
    print("-" * 40)

def main():
    """Interactive database query interface."""
    print("=== Quiz Smith Database Query ===\n")
    
    # Check API key
    if not hf_api_key:
        print("❌ HF_API_KEY not found. Please add it to your .env file.")
        return
    
    while True:
        print("\n" + "-"*50)
        print("Enter your search query (or 'quit' to exit):")
        user_query = input("> ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_query:
            print("❌ Please enter a valid query.")
            continue
        
        # Ask for number of chunks
        try:
            num_chunks_input = input("Number of chunks to retrieve (default: 5): ").strip()
            num_chunks = int(num_chunks_input) if num_chunks_input else 5
        except ValueError:
            num_chunks = 5
        
        # Search database
        content, docs = search_database(user_query, num_chunks)
        
        # Display results
        display_retrieved_content(content, docs)

if __name__ == "__main__":
    main()