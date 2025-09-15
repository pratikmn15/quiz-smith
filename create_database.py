# Data Ingestion using LangChain from PDF source documents (HuggingFace API embeddings)
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Make sure your HF_API_KEY is in a .env file
hf_api_key = os.getenv("HF_API_KEY")

def test_huggingface_api():
    """Test if HuggingFace API is working properly."""
    print("Testing HuggingFace API connection...")
    
    if not hf_api_key:
        print("❌ HF_API_KEY not found in environment variables.")
        print("Please add your HuggingFace API key to the .env file:")
        print("HF_API_KEY=your_api_key_here")
        return False
    
    try:
        # Test embeddings with a simple text using the new class
        embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-base-en-v1.5",
            huggingfacehub_api_token=hf_api_key
        )
        
        # Test with a simple sentence
        test_text = "This is a test sentence to verify the API connection."
        test_embedding = embeddings.embed_query(test_text)
        
        if test_embedding and len(test_embedding) > 0:
            print(f"✅ HuggingFace API is working!")
            print(f"   Model: BAAI/bge-base-en-v1.5")
            print(f"   Embedding dimension: {len(test_embedding)}")
            return True
        else:
            print("❌ HuggingFace API returned empty embedding.")
            return False
            
    except Exception as e:
        print(f"❌ HuggingFace API error: {str(e)}")
        print("Please check:")
        print("1. Your API key is valid")
        print("2. You have internet connection")
        print("3. The model BAAI/bge-base-en-v1.5 is accessible")
        return False

def create_pdf_database():
    # First, test the API
    if not test_huggingface_api():
        print("\n❌ Cannot proceed without working HuggingFace API.")
        return None
    
    print("\n" + "="*50)
    print("Starting PDF database creation...")
    print("="*50)
    
    # 1. Data Ingestion - Load all PDF files from source_documents directory
    source_dir = "data"
    
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Created directory: {source_dir}")
        print("Please add PDF files to the source_documents directory.")
        return None
    
    pdf_files = glob.glob(os.path.join(source_dir, "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in source_documents directory.")
        return None
    
    print(f"Found {len(pdf_files)} PDF files.")
    
    # Load all PDF documents
    all_documents = []
    for pdf_file in pdf_files:
        print(f"Loading {os.path.basename(pdf_file)}...")
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"✓ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"❌ Error loading {pdf_file}: {str(e)}")
    
    if not all_documents:
        print("❌ No documents were successfully loaded.")
        return None
    
    print(f"Total documents loaded: {len(all_documents)}")
    
    # 2. Preprocessing & Chunking - Use RecursiveCharacterTextSplitter
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    text_chunks = text_splitter.split_documents(all_documents)
    print(f"✓ Split into {len(text_chunks)} chunks with 50 token overlap.")
    
    # 3. Indexing (Vector Store) - Create embeddings using Hugging Face API
    print("\nCreating vector store with embeddings...")
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-base-en-v1.5",
            huggingfacehub_api_token=hf_api_key
        )
        
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            collection_name="pdf_collection",
            persist_directory="./chroma_db"
        )
        
        print("✓ Created Chroma vector store with text chunks.")
        
        # Persist to disk
        vectorstore.persist()
        print("✓ Persisted vector store to ./chroma_db directory.")
        
        print("\n" + "="*50)
        print("✅ Database creation completed successfully!")
        print(f"📊 Total chunks indexed: {len(text_chunks)}")
        print(f"🗃️  Collection name: pdf_collection")
        print(f"💾 Persist directory: ./chroma_db")
        print("="*50)
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return None

if __name__ == "__main__":
    create_pdf_database()
