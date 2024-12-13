import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import tempfile

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

# if __name__ == "__main__":
#     tokenizer, model, device = load_quantized_model()
#     prompt = "The capital of France is"
#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding=True,
#         truncation=True
#     ).to(device)

#     outputs = model.generate(
#         inputs.input_ids,
#         attention_mask=inputs.attention_mask,  # Provide attention mask
#         max_length=20,
#         temperature=0.7,
#         do_sample=True,  # Enable sampling
#         pad_token_id=tokenizer.eos_token_id  # Explicitly set pad token
#     )

#     print(tokenizer.decode(outputs[0], skip_special_tokens=True))

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
# if __name__ == "__main__":
#     index = setup_index()
#     add_documents_to_index(index, ["Paris is the capital of France.", "Berlin is the capital of Germany."])
#     query = "capital of France"
#     results = retrieve_documents(index, query)
#     print("Retrieved Documents:", results)

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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs.input_ids, max_length=100, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    rag_model = RAGModel()
    rag_model.add_to_corpus(["Rome is the capital of Italy.", "Madrid is the capital of Spain."])
    results = rag_model.retrieve_documents("capital of Spain")
    print("Retrieved Documents:", results)

    prompt = "What is the capital of Spain?"
    tokenizer, model, device = rag_model.tokenizer, rag_model.model, rag_model.device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=20,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id  # Use explicitly set pad token
    )

    print("Generated Text:", tokenizer.decode(outputs[0], skip_special_tokens=True))


