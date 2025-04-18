nsys profile \
  -o nsys_out/llama3_8b_profile_stats.qdrep \
  --trace=cuda,nvtx,cublas,cudnn \
  --stats=true \
  --sample=none \
  --force-overwrite=true \
  python llama3_sing.py