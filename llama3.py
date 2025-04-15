import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    model_path = "/data/models/Meta-Llama-3-8B"
    model = AutoModelForCausalLM.from_pretrained(
        "/data/models/Meta-Llama-3-8B",
        device_map={
            "": local_rank,
            "lm_head": local_rank
        },
        torch_dtype=torch.bfloat16
    ).eval()
    
    if local_rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer("The future of AI will be", return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=100)
        print(tokenizer.decode(outputs[0]))
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()