import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria
from sentence_transformers import SentenceTransformer, util
import tempfile
import argparse

class DenseRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading dense retriever model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.corpus_embeddings = None

    def add_documents(self, documents):
        self.corpus.extend(documents)
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)

    def retrieve(self, query, top_k=3):
        if self.corpus_embeddings is None:  # Explicitly check if embeddings are None
            raise ValueError("Corpus is empty. Add documents before retrieval.")
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)
        return [self.corpus[idx] for idx in top_results.indices]

def load_quantized_model(model_name="google/flan-t5-small"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading language model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    return tokenizer, model, device

class RAGModel:
    def __init__(self, model_name="google/flan-t5-small", retriever_model="all-MiniLM-L6-v2"):
        self.tokenizer, self.model, self.device = load_quantized_model(model_name)
        self.retriever = DenseRetriever(retriever_model)

    def add_to_corpus(self, documents):
        self.retriever.add_documents(documents)

    def retrieve_documents(self, query, top_k=3):
        return self.retriever.retrieve(query, top_k)

    def generate_response(self, query, top_k=3):
        retrieved_docs = self.retrieve_documents(query, top_k)
        context = "\n".join(retrieved_docs)
        prompt = f"""
        You are a helpful assistant providing factual answers based on the given context.

        Context:
        {context}

        Example Format:
        Answer: The capital of France is Paris. Paris is the largest city in France and its political, economic, and cultural center.
        Answer: The capital of Germany is Berlin. Berlin is known for its history, culture, and role in European politics.

        Please answer the following question based on the context and example format above.
        Question: {query}

        Answer:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # class StopAtAnswer(StoppingCriteria):
        #     def __init__(self, stop_token):
        #         super().__init__()
        #         self.stop_token = stop_token

        #     def __call__(self, input_ids, scores, **kwargs):
        #         decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        #         return self.stop_token in decoded_text

        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            num_beams=5,  # Beam search for deterministic answers
            early_stopping=True,
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
# pip install transformers sentence-transformers torch
# echo "Setup complete. Run the program with: python <script_name>.py --help"
# ```