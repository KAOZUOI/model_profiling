import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/models/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).eval()
input_text = "The future of AI is"
inputs = tokenizer(
    input_text, 
    return_tensors="pt"
).to("cuda:0")

nvtx.range_push("Full_Inference")
with torch.no_grad():
    outputs = model(**inputs)
    torch.cuda.synchronize()
nvtx.range_pop()

result = tokenizer.decode(outputs.logits.argmax(-1)[0], skip_special_tokens=True)
print(result)