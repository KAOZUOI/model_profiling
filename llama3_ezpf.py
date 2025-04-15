import torch.profiler
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

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with torch.no_grad():
        output = model(**inputs)
    
result = tokenizer.decode(output.logits.argmax(-1)[0], skip_special_tokens=True)
print(result)
prof.export_chrome_trace("trace.json")