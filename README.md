# 📚 RAG Pipeline: Chat with your Documents

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12-blue)

This repository contains a modular implementation of a Retrieval-Augmented Generation (RAG) pipeline using FastAPI. The pipeline includes components for data loading, chunking, embedding generation, vector storage, reranking, and LLM inference.

## Table of Contents
- [⚙️ Installation & Usage](#installation)
- [🧩 Modules](#modules)
- [🤝 Contributing](#contributing)
- [📝 License](#license)
- [🧑‍💻 Author](#author)

## ⚙️ Installation & Usage
1. Clone the repository
```bash
git clone https://github.com/eduquintanillae/rag-pipeline.git
cd rag-pipeline
```
2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages
```bash
pip install -r requirements.txt
```
4. Create a `.env` file in the root directory and set the `OPENAI_API_KEY` variable with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```
5. Run uvicorn to start the FastAPI server
```bash
uvicorn main:app --reload
```

6. Send a request to the POST /chat/completion endpoint using requests in Python:

<details> <summary>	 Click to expand Python snippet</summary>

```python
import requests

url = "http://localhost:8000/chat/completion"

data = {
    "method": "your_method",
    "model_name": "your_model",
    "n_questions_per_chunk": 5,
    "chunk_size": 500,
    "words_per_chunk": 100,
    "sentences_per_chunk": 3,
    "delimiter": "\n",
    "tokens_per_chunk": 512,
    "semantic_clusters": 10
}

response = requests.post(url, files=files, data=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
```
</details>


## 🧩 Modules

This project is structured into several key modules, each responsible for a specific part of the RAG pipeline:

### Data Loader

The Data Loader module is responsible for loading data from various sources, such as text files, PDFs, and Word documents. It preprocesses the data to ensure it's in a suitable format for chunking and labeling.

### Data Chunker

The Data Chunker module takes the preprocessed data and divides it into smaller, manageable chunks. This is essential for training LLMs, as they often have limitations on the maximum input size.

The Chunker has several strategies for chunking, including:
- **character**: Splits the text into chunks of a specified number of characters.
- **word**: Splits the text into chunks of a specified number of words.
- **sentence**: Splits the text into chunks based on sentence boundaries.
- **paragraph**: Splits the text into chunks based on paragraph boundaries.
- **delimiter**: Splits the text into chunks based on a specified delimiter (e.g. '\n').
- **token**: Splits the text into chunks based on a specified number of tokens.
- **semantic**: Splits the text into chunks based on semantic meaning.

### Embedding Generator

The Embedding Generator module generates embeddings for the text chunks using a specified model. This is crucial for semantic search and retrieval tasks, as it allows the system to understand the meaning of the text.

### Vector Storage

The Vector Storage module stores the generated embeddings in a vector database. This allows for efficient retrieval of relevant chunks based on semantic similarity.

### Reranker

The Reranker module takes the retrieved chunks and ranks them based on their relevance to the user's query. This ensures that the most relevant information is presented first. 

### Prompt Constructor

The Prompt Constructor module is responsible for constructing the prompts for the LLM based on the retrieved contexts and user query.

### LLM Inference

The LLM Inference module uses a large language model (LLM) to generate responses based on the ranked chunks. It handles user queries and provides answers based on the retrieved information.

### Pipeline Manager

The Pipeline Manager module orchestrates the entire RAG pipeline, coordinating the flow of data between the various components. It ensures that each step is executed in the correct order and that the necessary inputs and outputs are handled appropriately.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request. See `CONTRIBUTING.md` for details.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🧑‍💻 Author
Created by [Eduardo Quintanilla](https://github.com/eduquintanillae) - feel free to reach out for any questions or suggestions.