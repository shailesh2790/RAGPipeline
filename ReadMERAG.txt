RAG (Retrieval-Augmented Generation) Pipeline Project
=====================================================

This project implements an RAG pipeline using Flask, Milvus (vector database), and Ollama (for language model generation).

Just so you know, I have uploaded 2 main files to run. The data was too large even after compressing so unable to send it through zip file.
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset,2400 resumes can be taken from the data folder containing all departments' resumes.Save it in the milvus_data folder under the same directory as python files. Code should run smoothly and Postman can be used to query the directory.

Setup and Installation
----------------------
1. Install required Python packages:
   pip install flask pymilvus torch transformers sentence-transformers requests

2. Install and start Milvus:
   - Follow Milvus installation guide: https://milvus.io/docs/install_standalone-docker.md
   - Start Milvus using Docker

3. Install and start Ollama:
   - Follow Ollama installation guide: https://github.com/jmorganca/ollama
   - Start Ollama and download the llama2 model:
     ollama run llama2

Preprocessing Steps
-------------------
1. Prepare your document corpus (e.g., PDF files) in a designated folder.I have used Kaggle dataset of 2400 resumes for this project.https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset.

2. Run the preprocessing script to:
   - Extract text from PDFs
   - Split text into chunks
   - Generate embeddings for each chunk
   - Store chunks and embeddings in Milvus

Details of retrieval methodology

1. Document Retrieval:
   - The system uses Python's os module to walk through the specified directory.
   - It looks for files with the .pdf extension in the designated folder and its subfolders.
   - Method: os.walk() is used to recursively explore the directory structure.

2. Text Extraction:
   - For each PDF file found, the system uses the pypdf library to extract text.
   - Each page of the PDF is processed, and the text is concatenated.

3. Text Chunking:
   - The extracted text is split into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
   - This ensures that the text chunks are of manageable size for embedding and retrieval.

4. Embedding Generation:
   - Each text chunk is converted into a vector embedding using a SentenceTransformer model.
   - The specific model used is 'all-MiniLM-L6-v2', which produces 384-dimensional embeddings.

5. Storage in Vector Database:
   - The text chunks and their corresponding embeddings are stored in Milvus.
   - Each record in Milvus contains the text content, its embedding, and the source file path.

Code Snippet for Document Retrieval:
```python
import os
import pypdf

def preprocess_pdfs(folder_path: str):
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    pdf = pypdf.PdfReader(f)
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                    documents.append({"content": text, "source": file_path})
    return documents
```

This method allows for:
- Recursive exploration of the specified directory and its subdirectories
- Processing of all PDF files found in these directories
- Extraction of text content from each PDF
- Association of the extracted text with its source file path


Vector Database (Milvus) Setup
------------------------------
1. Ensure Milvus is running (usually on localhost:19530)
2. The RAG pipeline will automatically:
   - Connect to Milvus
   - Create a collection for storing document chunks and their embeddings
   - Insert preprocessed data into the collection

Running the API
---------------
1. Start the Flask application:
   python app.py
2. The API will be available at http://localhost:5000

API Endpoints
-------------
- GET /: Returns welcome message and usage instructions
- POST /query: Accepts JSON with a 'query' field, returns RAG-generated response

Testing API using Postman
-------------------------
1. Open Postman
2. Create a new POST request
3. Set the URL to http://localhost:5000/query
4. In the Headers tab, add:
   Key: Content-Type
   Value: application/json
5. In the Body tab:
   - Select 'raw'
   - Choose 'JSON' from the dropdown
   - Enter your query in this format:
     {
       "query": "Your question here"
     }
6. Click 'Send' to submit the request
7. The response will appear in the lower part of the Postman window

Troubleshooting
---------------
- Ensure Milvus and Ollama are running
- Verify that the llama2 model is available in Ollama.

Note: This README assumes you have Python, Docker, and necessary dependencies installed on your system.
