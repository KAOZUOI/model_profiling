ncu \
  --kernel-name "nvjet_tst_64x8_64x16_2x1_v_bz_TNT" \
  --set detailed \
  --section SpeedOfLight \
  --export ncu_out/roofline \
  --force-overwrite \
  --launch-skip 0 \
  --launch-count 1 \
  python llama3_sing.py