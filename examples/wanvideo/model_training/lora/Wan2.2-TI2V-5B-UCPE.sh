# Wan2.2 TI2V-5B I2V LoRA + additive UCPE ray conditioning.
# Run from the repository root. Set DATASET_PATH to your own dataset.

DATASET_PATH="${DATASET_PATH:-data/my_ucpe_dataset}"
OUTPUT_PATH="${OUTPUT_PATH:-models/train/Wan2.2-TI2V-5B_ucpe_lora}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  # This file must contain both DiT LoRA and ucpe_camera_encoder.* weights.
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_PATH" \
  --dataset_metadata_path "$DATASET_PATH/metadata.csv" \
  --data_file_keys "video,camera_trajectory" \
  --extra_inputs "input_image,camera_trajectory" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --ucpe_camera_condition "relray_absmap" \
  --ucpe_camera_hidden_dim 256 \
  --ucpe_camera_dropout 0.0 \
  --enable_csv_log \
  "${EXTRA_ARGS[@]}"

# For 121-frame training, change --num_frames to 121 and ensure each camera
# trajectory has frame indices 0..120. Ray temporal positions are 0,4,...,120.
# To continue an old pure-LoRA run, use --lora_checkpoint OLD_LORA.safetensors;
# the camera encoder starts at zero and is trained from scratch.
