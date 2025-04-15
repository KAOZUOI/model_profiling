import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/models/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=None,
    low_cpu_mem_usage=True
).eval().to('cuda:0')

input_text = "The future of AI is"
inputs = tokenizer(
    input_text, 
    return_tensors="pt"
).to('cuda:0')

nvtx.range_push("Full_Inference")
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=100
    )
nvtx.range_pop()
print(tokenizer.decode(outputs[0]))