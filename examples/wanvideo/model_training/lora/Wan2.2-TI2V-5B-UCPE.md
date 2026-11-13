# Wan2.2 TI2V-5B I2V LoRA + UCPE ray

This is a **UCPE-representation adaptation**, not a direct port of UCPE's
per-block camera self-attention. It preserves the geometric Relative Ray
Encoding (`world_to_ray_mats`) and optional absolute up/latitude map, then
projects them once and adds them to Wan's existing patch embeddings.

- Base model: Wan2.2 TI2V-5B, native fused first-frame I2V conditioning.
- No extra tokens, ControlNet, per-block camera attention, depth, or 3D warping.
- The camera embedding is computed once per training sample/inference call and
  shared by all denoising steps and both CFG branches. DiT attention cost is
  unchanged; the small ray construction/projection is additional overhead.
- Supported here: `relray_absmap` (19 channels, recommended) and `relray` (16).
- Not supported: sliding-window denoising and other Wan architectures. USP has
  not been validated; start with ordinary DDP/single-GPU training.

Reference: https://github.com/chengzhag/UCPE

## 1. Data

For a step-by-step raw-data conversion specification written for another AI to
execute, including camera/video synchronization, intrinsics after cropping,
quality checks and validation commands, see
`Wan2.2-TI2V-5B-UCPE-data-preparation.zh-CN.md`.

Start with your existing Wan2.2 TI2V-5B dataset and add one metadata column:

```csv
video,prompt,camera_trajectory
videos/0001.mp4,"a moving car",cameras/0001.json
```

Set `DATASET_PATH` and run:

```bash
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-UCPE.sh
```

The script keeps the original LoRA target modules/rank and enables:

```text
--data_file_keys video,camera_trajectory
--extra_inputs input_image,camera_trajectory
--ucpe_camera_condition relray_absmap
```

Each video must have at least the requested prefix of frames. The loader takes
the first `--num_frames` video frames, up to the available length, then shortens
to `4N+1`; this initial integration does not perform FPS resampling. Camera
`frame_index` must refer to that sampled RGB sequence.

### Recommended JSON schema

`c2w` means camera-to-world. OpenCV coordinates are **x right, y down, z
forward**. `image_size` is the original video size **before** DiffSynth's crop
and resize. Intrinsics are in original pixels: `(fx, fy, cx, cy)`.

```json
{
  "coordinate_system": "opencv",
  "image_size": [832, 480],
  "intrinsics": [600.0, 600.0, 416.0, 240.0],
  "translation_scale": 1.0,
  "frames": [
    {
      "frame_index": 0,
      "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    },
    {
      "frame_index": 1,
      "c2w": [[1, 0, 0, 0.00625], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    }
  ]
}
```

See `ucpe_camera_example/camera_right.json` for all 49 frames. That example is
a format demonstration, not a trajectory paired with an actual training video.
Use the real camera parameters for each video when training.

Accepted alternatives:

- Top-level `c2w` or `w2c` arrays, shaped `[T,4,4]` or `[T,3,4]`.
- `intrinsics` as one `[fx,fy,cx,cy]`, one value per frame `[T,4]`, or K `[T,3,3]`.
- `intrinsics_normalized: true` for width/height-normalized intrinsics. Supply
  `image_size` if crop/resize changes the aspect ratio.
- `coordinate_system: "opengl"` (x right, y up, z backward), converted to OpenCV.
- UCPE-style `x_fov` in degrees plus `xi` (default 0), scalar or per frame. This
  uses the original UCM focal formula and square-pixel assumption. If cover
  resizing would change the aspect ratio, convert these to pixel intrinsics;
  direct UCM cropping is rejected.
- `.jsonl` (one frame object per line), `.npz`, and RealEstate10K `.txt`.
  RealEstate10K files contain normalized intrinsics and world-to-camera poses.
  Their timestamp column is not a frame index: camera rows are matched to the
  RGB prefix in row order, without FPS resampling. Preprocess/resample the video
  and camera rows together first if the original timestamps are irregular.

The code applies the same resize-to-cover and center-crop geometry as
`ImageCropAndResize` to `fx/fy/cx/cy`. Do not disable this unless your K has
already been transformed; in that case set `intrinsics_already_cropped: true`.
For missing sampled poses, invalid UCM parameters, or ambiguous normalized
intrinsics, loading fails rather than silently repeating a pose.

### Alignment and scale

For `F` RGB frames, Wan2.2 produces `(F-1)/4+1` latent temporal positions:

| RGB frames | Ray frames | Ray source indices | Ray spatial grid at 480x832 |
|---|---:|---|---:|
| 49 | 13 | `0,4,...,48` | `15 x 26` |
| 121 | 31 | `0,4,...,120` | `15 x 26` |

The `15 x 26` grid follows `16x16 VAE compression x 2x2 DiT patches` = a 32x32
pixel token footprint. By default all poses are transformed by the inverse of
the first selected camera pose. This is appropriate for first-frame I2V. Use
`--ucpe_camera_world_coordinates` only if you deliberately train with absolute
world coordinates and use `--world_coordinates` at inference.

Camera translations must have a consistent scale. `translation_scale` in each
file and `--ucpe_camera_translation_scale` are multiplied together. UCPE's
PanShot preprocessing divides translation by near depth; if you follow that
scheme, set per-file `translation_scale` to `1 / near_depth` (the depth model
itself is not run here). Keep the same normalization in training and inference.
Absolute scale cannot be inferred from pixels alone.

## 2. Checkpoints and resuming

The output `epoch-N.safetensors` contains both:

- standard DiT `lora_A/lora_B` tensors;
- full `ucpe_camera_encoder.*` tensors (zero-initialized output projection).

This training entry requires the Wan2.2-5B 30-layer DiT and all six LoRA
targets (`q,k,v,o,ffn.0,ffn.2`) in **every** layer. A UCPE checkpoint containing
DiT LoRA tensors is checked for all 360 LoRA A/B tensors before inference
fusion; an incomplete checkpoint is rejected. A camera-only warm-start file may
omit LoRA tensors.

Resume **weights** for both branches:

```bash
RESUME_FROM_CHECKPOINT=models/train/Wan2.2-TI2V-5B_ucpe_lora/epoch-1.safetensors \
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-UCPE.sh
```

`--resume_from_checkpoint` in this framework restores model weights only; it
does not restore optimizer, epoch, or RNG state.

To initialize from an older **pure DiT LoRA**, use `--lora_checkpoint OLD.safetensors`.
Do not use that option to restore a UCPE camera checkpoint: the camera encoder
is not encoded in DiT LoRA tensors.

## 3. Inference

```bash
python examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B-UCPE.py \
  --checkpoint models/train/Wan2.2-TI2V-5B_ucpe_lora/epoch-4.safetensors \
  --camera_trajectory path/to/target_camera.json \
  --input_image path/to/first_frame.png \
  --prompt "your prompt" \
  --height 480 --width 832 --num_frames 49 \
  --cfg_scale 5 --num_inference_steps 50 --fps 24 \
  --output wan22_ucpe.mp4
```

The inference script infers ray condition and hidden dimension from the
checkpoint, strictly loads camera weights, then loads the DiT LoRA. It applies
the same cover-resize/center-crop to the first-frame image as the training data.
Local model paths can be supplied as JSON with `--model_paths`. The target trajectory must
cover the requested `num_frames`; it need not be the training trajectory.

## 4. Practical notes

- `--ucpe_camera_dropout 0.1` is optional regularization. It drops camera input
  only during training; inference does not add a separate camera-unconditional
  DiT pass. Both text CFG branches retain the same camera condition.
- A zero-initialized final projection gives an exact no-camera cold start.
- This implementation has not been pretrained or quality-validated on your
  dataset. Useful training data needs accurate pose/intrinsics alignment and
  should include object motion under a static camera, not only static scenes
  with moving cameras. Otherwise the model may suppress subject motion.
- Official UCPE weights/configurations target other Wan backbones and are not
  directly loadable as Wan2.2 TI2V-5B weights. Train this camera encoder and
  DiT LoRA for this base model.
