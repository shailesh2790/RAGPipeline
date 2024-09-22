# preprocess_data.py

import os
import logging
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Update the path to your data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "milvus_data", "data")

def preprocess_pdfs(folder_path: str = DATA_DIR) -> List[Dict[str, str]]:
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        pdf = pypdf.PdfReader(f)
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text()
                        documents.append({"content": text, "source": file_path})
                    logging.info(f"Processed PDF: {file_path}")
                except Exception as e:
                    logging.error(f"Error processing PDF {file_path}: {str(e)}")
    return documents

def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, str]]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc['content'])
        chunks.extend([{"content": chunk, "source": doc['source']} for chunk in doc_chunks])
    logging.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def generate_embeddings(chunks: List[Dict[str, str]]) -> List[Dict[str, any]]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for chunk in chunks:
        chunk['embedding'] = model.encode(chunk['content']).tolist()
    logging.info(f"Generated embeddings for {len(chunks)} chunks")
    return chunks

def setup_milvus_collection(collection_name: str = "resume_chunks"):
    try:
        connections.connect("default", host="localhost", port="19530")
        logging.info("Connected to Milvus")
        
        if utility.has_collection(collection_name):
            logging.info(f"Collection '{collection_name}' already exists. Using existing collection.")
            return Collection(collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000)
        ]
        schema = CollectionSchema(fields, "Resume chunks collection")
        collection = Collection(collection_name, schema)
        
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        logging.info(f"Created new collection '{collection_name}' and index")
        return collection
    except Exception as e:
        logging.error(f"Error setting up Milvus collection: {str(e)}")
        raise

def insert_into_milvus(collection, chunks: List[Dict[str, any]]):
    try:
        entities = [
            [chunk['content'] for chunk in chunks],
            [chunk['embedding'] for chunk in chunks],
            [chunk['source'] for chunk in chunks]
        ]
        collection.insert(entities)
        collection.flush()
        logging.info(f"Inserted {len(chunks)} entities into Milvus")
    except Exception as e:
        logging.error(f"Error inserting into Milvus: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Data preprocessing and database setup
        documents = preprocess_pdfs()
        chunks = chunk_documents(documents)
        embedded_chunks = generate_embeddings(chunks)
        
        collection = setup_milvus_collection()
        insert_into_milvus(collection, embedded_chunks)
        
        logging.info("Data preprocessing and Milvus insertion complete.")
    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")