import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_quantized_model(model_name="gpt2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)

    return tokenizer, model, device

if __name__ == "__main__":
    tokenizer, model, device = load_quantized_model()
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_length=20, temperature=0.7)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
