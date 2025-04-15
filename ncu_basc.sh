ncu \
  --target-processes all \
  --kernel-regex "gemm|attention" \
  --metrics smsp__sass_thread_inst_executed_op_ffma.sum \
  --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
  python llama3_sing.py