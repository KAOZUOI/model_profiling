ncu \
  --kernel-name "nvjet_tst_64x8_64x16_2x1_v_bz_TNT" \
  --metrics "smsp__sass_thread_inst_executed_op_ffma.sum,smsp__pipe_tensor_cycles_active.avg,dram__bytes.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum" \
  --section SpeedOfLight \
  --export roofline \
  --force-overwrite \
  python llama3_sing.py