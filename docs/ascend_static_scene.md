# 910B2 静态场景相机控制微调与推理

## 1. 目标与边界

目标是**静态场景相机重拍摄**：只改变视角，人物姿态、表情、物体状态和背景内容不自主变化。相机导致的正常遮挡变化、透视和视差必须保留，不能把所有像素运动都当成错误。

本扩展提供：

- 单卡 Ascend 910B2 的 NPU 设备、BF16、HCCL（仅多卡时）和 CPU offload；
- 冻结 UMT5、CLIP、Wan VAE 的特征缓存；
- 冻结 Wan2.1-14B 主干，默认只训练 Uni3C 控制器 LoRA；
- 保留 `render.mp4`、`render_mask.mp4`、`cam_info.json` 和 ray embedding；
- 在 `cam_control.py` 中显式加载静态场景 LoRA；
- CUDA 单卡回退路径。

**当前环境没有 910B2、CANN、PyTorch、diffusers、PEFT、真实模型权重或训练数据，因此只完成 Python AST 与差异/空白静态检查；没有执行张量数值测试、PEFT 保存/加载测试或端到端训练。未宣称已在昇腾上跑通、收敛或达到静态效果。** 第一次运行必须按第 8 节做 smoke test。

`alignment.py`/`uni3c_inference.py` 是人体动作驱动入口，目标是让人物跟随 SMPL 动作，**不是本任务的静态人物入口**。本次训练与静态推理由 `prepare_static_cache.py`、`train_static_lora.py` 和 `cam_control.py` 组成。

## 2. 为什么需要静态场景数据

**准备数据前必须先阅读 [静态场景训练数据处理、坐标系与验收规范](static_scene_data.md)。** 该文档给出完整目录、`cam_info.json` schema、w2c/c2w 和世界/相机/像素约定、外部标定转换公式、主体坐标、真实轨迹渲染流程及逐项验收清单。本节只保留训练前的核心约束。

原始 Wan I2V 会生成人物动作，Uni3C 的点云控制主要约束相机和场景几何，不等于冻结主体。仅在 prompt 中写“不要动”通常不能稳定解决。

训练样本必须满足：

1. 同一场景有多个视角，场景中的物体和人物本身不动；
2. `reference_image` 是轨迹第一帧；
3. `target_video` 是真实静态场景沿目标相机轨迹拍摄的视频，而不是 Uni3C 自己生成的视频；
4. `render_path` 中的 render/mask/cam_info 对应同一参考图和同一目标轨迹；
5. target、render、mask 的帧数、顺序、分辨率和相机轨迹一致。

**相机对齐是可训练性的前提。** `cam_render.py` 生成的是指定虚拟轨迹，不会自动从 `target_video` 恢复真实相机。因此，不能拿一段任意真实环绕视频，再用随手指定的 `--d_phi` 生成 render 就开始训练。应使用已知标定轨迹/同步多相机，或先用可靠的 SfM/SLAM 标定恢复逐帧 w2c 和 intrinsic，并将它们转换到 Uni3C 的相机坐标约定、导出相同 `cam_info.json` 后渲染。当前脚本尚未实现“从 target_video 自动估计轨迹并写入 cam_info”的入口；对未标定视频先人工检查 render 与 target 的透视、主体尺度、遮挡顺序和首帧对齐。

可使用静态多视角/室内扫描数据，或自己用滑轨、机械臂、多相机采集。雕像、人体模型、保持静止的人可以作为静止主体，但要剔除眨眼、呼吸导致的大幅姿态变化、走动、说话、衣物明显运动、树木摆动、水面和动态光影。普通人物跳舞/走路视频不能作为本任务的正样本。

建议先从 500～5000 个短片段开始；数据质量比单纯数量更重要。训练集要覆盖人物、物体、室内、室外、近远景以及环绕/俯仰/推拉轨迹。另留 20～50 个场景做验证，不能与训练场景重复。

### 不需要的错误监督

- 不要简单惩罚“输出每帧与参考图的像素差”：这会抑制正确相机视差。
- 不要用带人物动作的视频配一个静止 prompt：模型仍会学到动作。
- 不要用当前 Uni3C 输出作为唯一 ground truth：会复制现有伪影。

## 3. 910B2 环境

昇腾官方版本矩阵中的一个可行组合是：

```text
Python 3.10
PyTorch 2.6.0
torch-npu 2.6.0.post5（TorchNPU 7.3.0）
CANN 8.5.0
与机器驱动/固件匹配的运行环境
```

也可以使用官方矩阵中的其他完整组合，但不要混装 CUDA wheel、PyTorch、torch_npu 和 CANN。先检查：

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 按实际安装路径修改
conda create -n uni3c-npu python=3.10 -y
conda activate uni3c-npu
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.6.0.post5
pip install -r requirements-ascend.txt
python -c "import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__); print(torch.npu.is_available()); print(torch.npu.get_device_name(0))"
```

不要在 NPU 环境安装 `xfuser[flash-attn]`。单卡代码路径不导入 xFuser；CUDA flash-attn wheel 不能用于昇腾。训练建议保留较大的主机内存和 swap，因为 14B 权重加载、CPU offload 和数据缓存都在 NPU 显存之外占用资源。

官方版本来源：

- [Ascend PyTorch 版本兼容矩阵](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
- [TorchNPU 安装与快速开始](https://github.com/Ascend/pytorch)
- [昇腾 CANN 下载](https://www.hiascend.com/cann/download)

## 4. 点云预处理

推荐沿用现有 GPU 工作流生成 `render_path`，然后把结果复制到 910B2 主机。NPU 训练和相机控制推理不需要 PyTorch3D。

如果只有 910B2 主机，可以尝试 CPU 预处理：

```bash
# 需要另行安装支持 CPU 的 PyTorch3D 和 CarveKit；MoGe-3 的 Triton/FlexGEMM 路径不是 CPU 回退。
python cam_render.py \
  --device cpu --depth_model depth-pro \
  --reference_image data/demo/flux-0.png \
  --output_path outputs/static_scene_000/render \
  --nframe 17 --traj_type custom --d_phi -15
```

这条路径也可能较慢，且本环境未验证 CPU PyTorch3D/CarveKit。不要把 `--device npu` 用在未经适配的 PyTorch3D 点云光栅化上。

## 5. 数据清单

创建 `data/static_train.jsonl`，每行一个样本；相对路径以 JSONL 所在目录为基准：

```json
{"sample_id":"scene000_cam01","reference_image":"samples/scene000/reference.png","render_path":"samples/scene000/render","target_video":"samples/scene000/target.mp4","prompt":"A completely static indoor scene; only the camera moves around the motionless person and furniture."}
{"sample_id":"scene001_cam02","reference_image":"samples/scene001/reference.png","render_path":"samples/scene001/render","target_video":"samples/scene001/target.mp4","prompt":"Static objects and background, rigid scene geometry, camera orbit only, no object motion."}
```

每个 `render_path` 包含：

```text
render.mp4
render_mask.mp4
cam_info.json
```

`cam_info.json` 的 `per_frame_intrinsic` 若存在，会与旧版单 `intrinsic` 一样被读取。官方配置启用 `camera_embedding`，缓存默认保存 ray embedding，训练与推理条件一致。

先用 `--nframe 17` 验证，再尝试 33/49/81。Wan 时序 VAE 要求 `nframe % 4 == 1`。目标视频必须提前裁成完全相同帧数，脚本不会静默抽帧或错位。

## 6. 缓存冻结特征

```bash
python prepare_static_cache.py \
  --device npu \
  --manifest data/static_train.jsonl \
  --output_dir cache/static_17 \
  --nframe 17 --max_area 368640 \
  --cache_dtype bfloat16
```

缓存包括：目标视频 latent、I2V condition、render latent/mask、ray embedding、UMT5 embedding 和 CLIP image embedding。通道契约如下（`F=(nframe-1)/4+1`）：

```text
target_latent / render_latent: [16,F,H/8,W/8]
condition:                    [20,F,H/8,W/8] = 4 mask + 16 image latent
Wan hidden_states:            [B,36,F,H/8,W/8] = 16 noisy + 20 condition
控制器卷积输入:               [B,36,F,H/8,W/8] = hidden_states[:,:20] + 16 render latent
render_mask:                  [1,nframe,H,W]
camera_embedding:             [6,nframe,H,W]（启用时）
```

缓存文件不含 batch 维；`DataLoader` 只添加一次 batch 维。缓存按样本单独保存，`.partial/` 保存尚未完成 VAE 编码的文本/CLIP 特征。`cache_manifest.json` 在启动大编码器前写入，记录设置、样本 ID、输入路径、prompt 和文件 SHA-256；改变配置、输入文件或样本集合后必须换输出目录。完整缓存通过校验才跳过；残缺缓存需使用 `--overwrite` 重建。最终 `.pt` 先写临时文件再原子替换，成功后删除对应 partial 文件。

`--cache_dtype bfloat16` 降低磁盘占用；需要精确复现 FP32 缓存时使用 `--cache_dtype float32`。若关闭 ray embedding，训练使用 `--no-camera_embedding`，推理也必须使用 `camera_embedding: false` 的 Uni3C 配置；默认配置两者均启用。

17 帧、约 480×768 的每个样本通常需要数百 MB，具体取决于尺寸和 ray embedding；先用 `du -sh cache/static_17` 检查磁盘。不要在训练时反复运行文本编码器、CLIP 或 VAE。

## 7. 单卡 LoRA 微调

```bash
python train_static_lora.py \
  --device npu \
  --cache_dir cache/static_17 \
  --output_dir outputs/static_lora \
  --target controlnet \
  --lora_rank 16 --lora_alpha 16 \
  --learning_rate 1e-5 \
  --max_steps 1000 \
  --batch_size 1 --gradient_accumulation 8 \
  --timestep_sampling logit_normal \
  --save_every 100
```

实现细节：

- Wan 主干和编码器冻结，默认仅控制器内部的 Linear 添加 LoRA；`controlnet_patch_embedding` 和 mask embedding 是 Conv3d，保持冻结；
- LoRA 参数和 AdamW 状态保留 FP32，主干计算使用 BF16；controller 卷积的 render/mask/ray 输入显式使用 FP32；
- 开启 Wan 和控制器 gradient checkpointing；diffusers 0.33.1 使用 `use_reentrant=False`，无需给冻结条件输入开启梯度；
- NPU 的 RoPE 在 CPU 构造频率网格并转为 complex64 后上传；主 Wan 和控制器 attention 都使用 shared complex64 RoPE，CUDA 保留原始 complex128 路径；
- 使用 flow-matching `x_t=(1-sigma)x_0+sigma*noise`，目标为 `noise-x_0`；
- sigma/timestep 使用与官方 Uni3C 配置一致的 `flow_shift=5`；
- 默认 logit-normal 时间步采样，可切换 `--timestep_sampling uniform`；
- 每 100 步保存 `step-xxxxxx` 和 `latest`，不会覆盖官方权重。

仅检查模型、LoRA、优化器、数据清单和一条缓存 batch 的形状（不执行前向/反向）：

```bash
python train_static_lora.py --device npu \
  --cache_dir cache/static_17 --output_dir outputs/validate \
  --validate_only
```

然后先把 `--max_steps 1 --gradient_accumulation 1 --save_every 1`，检查前向、反向和保存，再长训。训练只缓存编码器输出，**没有对 14B transformer 做训练时 CPU offload**：BF16 主干权重本身约 28 GB，controller、LoRA、激活和 AdamW 还需要额外显存/内存，不能仅凭“冻结主干”认定 910B2 一定装得下。CPU offload 当前用于单卡推理。

若 OOM：

1. 保持 `--batch_size 1`，增加梯度累积而不是扩大 batch；
2. 用 17 帧和更小 `--max_area` 重新缓存；
3. 保持 `--target controlnet`，不要先用 `--target all`；
4. 确认 gradient checkpointing 未被 `--no_gradient_checkpointing` 关闭；
5. 检查 SDPA 是否调用支持该形状的 NPU 融合实现；若落到显存与序列长度平方成正比的实现，当前脚本不能保证跑通长视频。

`--target all` 会额外训练 Wan 注意力 LoRA，可能提升适应能力，但激活、LoRA 梯度和优化器状态也更多，不是单卡默认。`--resume_lora outputs/static_lora/latest` 只恢复 LoRA 权重，**不恢复 optimizer/step**，因此不是严格断点续训；学习率和步数需按新阶段设置。保存只包含 adapter 张量和配置，不包含 Wan 14B 基座权重。

在完整 Ascend 环境中可以先运行轻量测试，再加载大模型：

```bash
python -m unittest tests.test_static_training -v
```

`tests/test_static_training.py` 包含真实 `DataLoader` 的 batch/通道契约测试；安装 `diffusers/peft/safetensors` 后还会运行一个小 Wan 模型的 adapter 保存、严格加载和前向一致性测试。本机缺少这些依赖，所以这些测试在这里**未执行**，不能把测试文件的存在视为通过证明。

## 8. 910B2 推理

```bash
python cam_control.py \
  --device npu \
  --reference_image data/static/samples/scene000/reference.png \
  --render_path data/static/samples/scene000/render \
  --output_path outputs/scene000_static.mp4 \
  --static_lora outputs/static_lora/latest \
  --lora_scale 1.0 \
  --nframe 17 --max_area 368640 \
  --prompt "Completely static scene. Only the camera moves; the person, objects and background remain motionless." \
  --seed 1024
```

基线对照只需去掉 `--static_lora`。建议固定同一参考图、轨迹、prompt 和 seed，对比：

- 是否准确沿目标相机轨迹运动；
- 人物是否出现走路、手势、表情、头部自主运动；
- 背景是否融化、漂移、凭空变化；
- 遮挡处补全是否稳定，主体身份和细边界是否保持。

LoRA 过强可能损害画质或运镜精度，可测试 0.5/0.75/1.0。**不能通过冻结输入 render 视频来宣称主体静止**；render 只是几何条件，最终视频仍由扩散模型生成。单目点云没有真实背面，大角度运镜仍需生成模型补全，不保证严格三维一致。

首次运行依次验证：

1. `torch.npu.is_available()` 与一次小张量乘法；
2. 缓存一个样本；
3. `--validate_only`；
4. 一个训练 step（前向、反向、保存）；
5. 官方基线推理；
6. LoRA 推理；
7. 再逐步增加帧数和分辨率。

记录 `npu-smi info`、CANN/torch/torch_npu 版本、峰值 HBM、主机内存、单步时间和 OOM 算子。当前代码不是经过真机性能调优的 MindSpeed/MindIE 工程，也没有把 PyTorch3D、Triton、flash-attn 或 NCCL 假装成昇腾原生能力。
