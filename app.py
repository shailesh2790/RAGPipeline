# app.py
import logging
import traceback
import requests
from flask import Flask, request, jsonify
from pymilvus import connections, Collection, DataType
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, collection_name: str = "resume_chunks"):
        try:
            connections.connect("default", host="localhost", port="19530")
            self.collection = Collection(collection_name)
            self.collection.load()
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("RAG Pipeline initialized")
            self.check_collection_schema()
        except Exception as e:
            logger.error(f"Error initializing RAG Pipeline: {str(e)}")
            raise

    def check_collection_schema(self):
        schema = self.collection.schema
        for field in schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                logger.info(f"Stored vector dimension: {field.params['dim']}")
                self.vector_dim = field.params['dim']
                break
        else:
            raise ValueError("No FLOAT_VECTOR field found in collection schema")

    def generate_query_embedding(self, query: str):
        embedding = self.sentence_transformer.encode(query)
        logger.info(f"Generated query embedding dimension: {len(embedding)}")
        if len(embedding) != self.vector_dim:
            raise ValueError(f"Query embedding dimension ({len(embedding)}) does not match stored dimension ({self.vector_dim})")
        return embedding.tolist()

    def retrieve(self, query: str, top_k: int = 5):
        try:
            query_embedding = self.generate_query_embedding(query)
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "source"]
            )
            logger.info(f"Retrieved {len(results[0])} chunks for query")
            return results[0]
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            raise

    def generate_response(self, query: str, retrieved_chunks):
        try:
            context = "\n".join([chunk.entity.get('content') for chunk in retrieved_chunks])
            prompt = f"""Use the following pieces of context to answer the query. If you cannot answer the query based on the given context, say "I don't have enough information to answer this query."

Context:
{context}

Query: {query}

Answer:"""
            
            logger.info("Sending request to Ollama API")
            response = requests.post('http://localhost:11434/api/generate', 
                                     json={
                                         "model": "llama2",
                                         "prompt": prompt,
                                         "stream": False
                                     })
            response.raise_for_status()
            logger.info("Received response from Ollama API")
            return response.json()['response']
        except requests.RequestException as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            raise

app = Flask(__name__)
rag_pipeline = None

@app.before_first_request
def initialize_rag_pipeline():
    global rag_pipeline
    try:
        rag_pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the RAG API",
        "usage": "Send a POST request to /query with a JSON payload containing a 'query' field"
    })

@app.route('/query', methods=['POST'])
def query_rag():
    if rag_pipeline is None:
        return jsonify({"error": "RAG pipeline not initialized"}), 500

    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query = data['query']
        logger.info(f"Received query: {query}")
        
        retrieved_chunks = rag_pipeline.retrieve(query)
        response = rag_pipeline.generate_response(query, retrieved_chunks)
        
        return jsonify({"response": response})
    except ValueError as ve:
        logger.error(f"Dimension mismatch error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True)