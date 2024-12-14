# Model for Bluestaq internship coding challenge

---
## **1. Environment Setup**

### Prerequisites
- Python 3.8 or later
- Virtual environment tool (e.g., `venv` or `conda`)

### **Automatic Setup**
Run the provided shell script to set up the environment and dependencies:

#### **Shell Script Contents**
```bash
#!/bin/bash
# virtual environment if necessary
python3 -m venv rag_env
source rag_env/bin/activate

# dependencies
pip install transformers sentence-transformers torch

echo "Setup complete. Run the program with: python rag_model.py --help"
```

### **Manual Setup**
1. Clone the repo and navigate to the project directory:
   git clone https://github.com/cdpearlman/BluestaqModel.git
   cd <repo-directory>
2. Use python3 environment or create a virtual environment (see above)
3. Install dependencies:
   pip install transformers sentence-transformers torch
---

## **2. Running the CLI Tool**

### **Command-line Arguments**
Run the program using:
python rag_model.py --corpus <documents> --query <query> --top_k <number>

- `--corpus`: Optional. A list of documents (strings) to add to the corpus (e.g., "Document 1" "Document 2").
- `--file`: Optional. A file path that contains a .txt file to add to the corpus (e.g., "test.txt")
- `--query`: The user’s query to ask the RAG model.
- `--top_k`: Optional. The number of top documents to retrieve (default is 3).

### **Examples**
#### Adding Documents to the Corpus
python rag_model.py --corpus "Madrid is the capital of Spain." "Rome is the capital of Italy."
python rag_model.py --file "example.txt"

#### Querying the Model
python rag_model.py --query "What is the capital of Spain?" --top_k 1

#### Combined Usage
python rag_model.py --corpus "Seville is known for its Gothic cathedral." 
                    --file "test.txt"
                    --query "What is the cultural significance of Seville?" --top_k 2

## **3. Architecture and Design**

### **Overview**

This system integrates:
1. **Dense Retrieval Mechanism**:
   - Uses a SentenceTransformer (`all-MiniLM-L6-v2`) to encode documents and queries into dense vectors.
   - Retrieves top-k most relevant documents using cosine similarity.

2. **Language Model**:
   - Utilizes a pre-trained generative language model (`google/flan-t5-small`) to process retrieved context and generate coherent answers.

### **Architecture Diagram**
```plaintext
+------------------+      +-----------------------+
|   User Query     | ---> |   Dense Retriever    |
+------------------+      +-----------------------+
                                 |
                                 v
                      +--------------------+
                      |  Retrieved Context |
                      +--------------------+
                                 |
                                 v
                  +----------------------------+
                  | Pre-trained Language Model |
                  +----------------------------+
                                 |
                                 v
                       +------------------+
                       |  Generated Answer |
                       +------------------+
```

### **Integration Details**
- Retrieved documents are combined into a prompt along with the user’s query.
- The language model generates an answer based on the query and context.
- A well-engineered prompt ensures coherent and factual responses.

---

## **5. Features and Future Directions**

### Current Features
- **Local Execution**: Fully functional on a standard laptop (16GB RAM, Intel i7).
- **Dense Retrieval**: Efficient and accurate document retrieval using semantic embeddings.
- **Generative Responses**: Generates coherent answers based on retrieved context.

### Future Enhancements
1. **Support for Larger Models**:
   - Integrate GPT-J or GPT-NeoX for more sophisticated responses.
   - Optimize memory usage for running larger models locally.
   - Will depend on speed as requirement by project

2. **Improved Retrieval**:
   - Add hybrid retrieval (combining dense and keyword-based approaches).
   - Incorporate document ranking metrics to refine results.

3. **Fine-tuned Language Model**:
   - Train the language model on domain-specific QA datasets to enhance relevance.
   - Will again depend on project requriements and specificity

4. **Evaluation Metrics**:
   - Quantitative benchmarks for response coherence, accuracy, and retrieval precision.

*NOTE* README and instructions created with the help of generative AI
---
