# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
python quick_eval_inst_small.py \
  --checkpoint_paths \
    /home/liuzhirui/model/Qwen3-VL-main/metaquery-main/checkpoints/output/qwen3vl2b_inst_small/checkpoint-370 \
    /home/liuzhirui/model/Qwen3-VL-main/metaquery-main/checkpoints/output/qwen3vl2b_t2i_small/checkpoint-930 \
  --device cuda \
  --dtype float16 \
  --num_inference_steps 30 \
  --guidance_scale 4.5 \
  --image_guidance_scale 1.5 \
  --seeds 42,43,44 \
  --output_dir ./quick_eval_outputs