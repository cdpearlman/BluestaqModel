import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import tempfile
import argparse

def load_quantized_model(model_name="gpt2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)

    return tokenizer, model, device

def setup_index():
    schema = Schema(content=TEXT(stored=True))
    index_dir = tempfile.mkdtemp()
    index = create_in(index_dir, schema)
    return index

def add_documents_to_index(index, documents):
    writer = index.writer()
    for doc in documents:
        writer.add_document(content=doc)
    writer.commit()

def retrieve_documents(index, query, top_k=3):
    searcher = index.searcher()
    query_parser = QueryParser("content", schema=index.schema)
    parsed_query = query_parser.parse(query)
    results = searcher.search(parsed_query, limit=top_k)
    return [result["content"] for result in results]

class RAGModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer, self.model, self.device = load_quantized_model(model_name)
        self.index = setup_index()

    def add_to_corpus(self, documents):
        add_documents_to_index(self.index, documents)

    def retrieve_documents(self, query, top_k=3):
        return retrieve_documents(self.index, query, top_k)
    
    def generate_response(self, query, top_k=3):
        retrieved_docs = self.retrieve_documents(query, top_k)
        context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuery: {query}\nAnswer: "

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="RAG Model CLI")
    parser.add_argument("--query", type=str, help="Query to ask the RAG model.")
    parser.add_argument("--corpus", type=str, nargs="*", help="Corpus documents to add.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve.")

    args = parser.parse_args()

    rag_model = RAGModel()

    if args.corpus:
        rag_model.add_to_corpus(args.corpus)

    if args.query:
        response = rag_model.generate_response(args.query, top_k=args.top_k)
        print("\nGenerated Response:\n", response)

if __name__ == "__main__":
    main()

# Shell script for setup (Linux):
# Save the following script as setup.sh for dependency installation
# ```bash
# #!/bin/bash
# python3 -m venv rag_env
# source rag_env/bin/activate
# pip install transformers whoosh
# echo "Setup complete. Run the program with: python <script_name>.py --help"
# ```