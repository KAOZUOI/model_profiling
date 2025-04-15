import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from accelerate import init_empty_weights, infer_auto_device_map
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    if world_size != 2:
        logger.warning(f"This script is designed for TP=2, but found {world_size} GPUs")

    model_path = "/data/models/Meta-Llama-3-8B"
    max_memory = {0: "18GiB", 1: "18GiB"}
    
    try:
        logger.info(f"[Rank {rank}] Loading model from local path: {model_path}")
        
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        
        max_memory_per_gpu = "18GiB"
        max_memory = {i: max_memory_per_gpu for i in range(world_size)}
        
        num_layers = getattr(config, "num_hidden_layers", 32)
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        
        device_map = {
            "model.embed_tokens": 0,
            "lm_head": 1,
            "model.norm": 1
        }

        for i in range(num_layers):
            device_map[f"model.layers.{i}"] = 0 if i < num_layers//2 else 1
        
        logger.info(f"[Rank {rank}] Using device map: {device_map}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if hasattr(model, "hf_device_map"):
            first_device = min([device for device in model.hf_device_map.values() 
                               if isinstance(device, int)])
        else:
            first_device = 0
        
        logger.info(f"[Rank {rank}] Preparing input")
        inputs = tokenizer(
            "The future of AI is", 
            return_tensors="pt"
        ).to(f"cuda:{first_device}")

        logger.info(f"[Rank {rank}] Warming up model...")
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        # 确保GPU同步
        torch.cuda.synchronize()
        dist.barrier()
        
        # 针对 LLM 推理的分段性能分析
        # 将编码和解码过程分开分析

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        with torch.no_grad():
            # 运行前向传播来编码输入
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 再运行一次
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # 准备解码输入（模拟生成的第一个 token）
        decoded_input = torch.cat([input_ids, torch.argmax(outputs.logits[:, -1:, :], dim=-1)], dim=1)

        with torch.no_grad():
            for _ in range(3):  # 3 次解码步骤，满足 warmup=1, active=2
                # 模拟单步解码
                decode_outputs = model(input_ids=decoded_input, use_cache=True)
                decoded_input = torch.cat([decoded_input, torch.argmax(decode_outputs.logits[:, -1:, :], dim=-1)], dim=1)


        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )

        if rank == 0:
            result = tokenizer.decode(outputs[0])
            logger.info(f"\nGenerated text: {result}")
            print(result)
            
    except Exception as e:
        logger.error(f"[Rank {rank}] Error during model loading or inference: {e}")
        import traceback
        logger.error(traceback.format_exc())
        dist.destroy_process_group()
        import sys
        sys.exit(1)

    logger.info(f"[Rank {rank}] Cleaning up")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()