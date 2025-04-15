import os
import torch
import torch.distributed as dist
import torch.profiler
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
        
        # 方法1: 使用auto_device_map实现TP=2
        # 先获取模型配置
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        
        # 使用accelerate推断最佳设备映射
        # 确保层被均匀分配到两张卡上
        max_memory_per_gpu = "18GiB"
        max_memory = {i: max_memory_per_gpu for i in range(world_size)}
        
        # 实现张量并行(TP=2)的设备映射
        # 现在使用加速器的自动推断，但给定更明确的约束
        # 将嵌入层放在第一张卡，输出层放在最后一张卡
        # 将transformer层平均分配到两张卡上
        num_layers = getattr(config, "num_hidden_layers", 32)  # 通常Llama3 8B有32层
        
        # 创建更智能的device_map - 实现真正的TP=2
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            
        # 明确的device_map
        device_map = {
            "model.embed_tokens": 0,  # 确保embedding层在GPU 0
            "lm_head": 1,  # 输出层在GPU 1
            "model.norm": 1
        }

        # 将Transformer层平均分配
        for i in range(num_layers):
            device_map[f"model.layers.{i}"] = 0 if i < num_layers//2 else 1
        
        logger.info(f"[Rank {rank}] Using device map: {device_map}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).eval()

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 获取模型实际使用的第一个设备
        if hasattr(model, "hf_device_map"):
            # 找出字典中最小的设备ID
            first_device = min([device for device in model.hf_device_map.values() 
                               if isinstance(device, int)])
        else:
            first_device = 0

        # 创建性能分析输出目录
        profile_dir = f"./logs/llama3_tp2_profile/rank{rank}"
        os.makedirs(profile_dir, exist_ok=True)
        
        # 输入处理
        logger.info(f"[Rank {rank}] Preparing input")
        inputs = tokenizer(
            "The future of AI is", 
            return_tensors="pt"
        ).to(f"cuda:{first_device}")

        # 预热模型 - 在进行性能分析前先预热
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

        # 1. 分析输入处理阶段
        logger.info(f"[Rank {rank}] Profiling input encoding phase...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            # 移除 tensorboard_trace_handler
            record_shapes=True,
            profile_memory=True
        ) as prof_encoding:
            # 第一阶段：对输入进行编码
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            with torch.no_grad():
                # 运行前向传播来编码输入
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 再运行一次
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # 直接导出为 Chrome Trace 格式
        prof_encoding.export_chrome_trace(f"{profile_dir}/encoding_trace.json")
        logger.info(f"[Rank {rank}] Encoding trace exported to {profile_dir}/encoding_trace.json")

        # 2. 分析模型单步解码（更接近真实 generate 行为）
        logger.info(f"[Rank {rank}] Profiling single decoding step...")
        # 准备解码输入（模拟生成的第一个 token）
        decoded_input = torch.cat([input_ids, torch.argmax(outputs.logits[:, -1:, :], dim=-1)], dim=1)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
            # 移除 tensorboard_trace_handler
            record_shapes=True,
            profile_memory=True
        ) as prof_decoding:
            with torch.no_grad():
                for _ in range(3):  # 3 次解码步骤，满足 warmup=1, active=2
                    # 模拟单步解码
                    decode_outputs = model(input_ids=decoded_input, use_cache=True)
                    decoded_input = torch.cat([decoded_input, torch.argmax(decode_outputs.logits[:, -1:, :], dim=-1)], dim=1)
                    prof_decoding.step()

        # 直接导出为 Chrome Trace 格式
        prof_decoding.export_chrome_trace(f"{profile_dir}/decoding_trace.json")
        logger.info(f"[Rank {rank}] Decoding trace exported to {profile_dir}/decoding_trace.json")

        # 3. 使用 CUDA 事件计时完整的 generate 过程
        logger.info(f"[Rank {rank}] Measuring complete generation time...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )

        end_event.record()
        end_time = time.time()
        torch.cuda.synchronize()

        inference_time = end_time - start_time
        cuda_time = start_event.elapsed_time(end_event) / 1000  # 转为秒
        logger.info(f"[Rank {rank}] Generation completed in {inference_time:.4f}s (wall time)")
        logger.info(f"[Rank {rank}] CUDA event time: {cuda_time:.4f}s")
        
        # 结果输出（仅主进程）
        if rank == 0:
            result = tokenizer.decode(outputs[0])
            logger.info(f"\nGenerated text: {result}")
            print(result)
            
            # 打印模型各部分所在设备的统计信息
            device_summary = {}
            for name, device in device_map.items():
                if device not in device_summary:
                    device_summary[device] = 0
                device_summary[device] += 1
            logger.info(f"Device allocation summary: {device_summary}")
            
    except Exception as e:
        logger.error(f"[Rank {rank}] Error during model loading or inference: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Clean up distributed process group before exit
        dist.destroy_process_group()
        import sys
        sys.exit(1)
    
    # Clean up
    logger.info(f"[Rank {rank}] Cleaning up")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()