# UCPE 训练原始数据处理说明书（供其他 AI 执行）

> **给执行本说明的 AI：** 你的任务是把用户的原始视频和真实相机参数转换为可训练的数据集，而不是仅解释本文。先盘点输入，再实施转换，最后实际读取并校验产物。不要修改或删除原始文件；不要凭空生成相机位姿、内参、视频帧或配对 prompt。无法确定坐标系、时间对应关系或裁剪方式时，先记录阻塞项并处理其他样本，不能用“看起来合理”的值补齐。

## 1. 目标与成功标准

训练入口是仓库根目录下的：

```bash
DATASET_PATH=/absolute/path/to/dataset \
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-UCPE.sh
```

脚本默认使用 **480×832、49 帧、Wan2.2 TI2V-5B 原生首帧 I2V、DiT LoRA + UCPE ray**。你的输出必须包含：

1. 可解码的训练视频；
2. 与视频逐帧对应的真实 `c2w` 相机轨迹；
3. 对应视频尺寸的像素内参；
4. 描述视频内容的 prompt；
5. `metadata.csv`，每行关联一个视频、一条 prompt 和一个相机 JSON；
6. 可复核的处理报告，列出成功、跳过、失败及待人工确认项。

`examples/wanvideo/model_training/lora/ucpe_camera_example/` 只是格式示例。不要把其中的合成轨迹当作真实训练样本。

---

## 2. 开始前必须确认的输入

先列出输入目录结构和少量样例，识别下列情况。不得假设所有原始文件采用同一种格式。

### A. 已有逐帧相机参数（优先）

输入至少应有：

- 视频，或有序 RGB 图像序列；
- 每帧位姿或可以按时间戳匹配的位姿；
- 相机内参，或可以可靠转换为像素内参的标定参数；
- 内参对应的原始图像分辨率；
- 坐标系和 `c2w` / `w2c` 含义；
- 帧号或时间戳，用于视频与相机同步。

先找现成的数据集说明、标定文件、COLMAP 输出、传感器日志或已有的处理脚本。优先沿用项目提供的、经过验证的转换逻辑，不要盲目重新实现坐标变换。

### B. 只有视频，没有相机参数

这不是可以直接运行的格式转换任务。不要把全单位矩阵、固定平移、Orbit 轨迹或随机轨迹作为真实标签。

需要额外的相机估计流程（例如适用于该数据类型的 SLAM/SfM/标定流程），输出带内参的逐帧位姿。选择工具时先检查场景是否适合：强动态物体、弱纹理、纯旋转、模糊、滚动快门和场景切换都可能导致失败。

估计完成后必须：

- 剔除跟踪丢失、重定位跳变、明显尺度漂移或内参不合理的片段；
- 抽帧投影或渲染可视化，检查相机运动方向、图像运动和焦距是否一致；
- 把估计尺度或归一化方式写入报告；
- 无法可靠验证的样本放入隔离目录，不进入正式训练集。

### C. 只有单张图像

单张图像不能生成一段真实相机监督视频。不要将静态图复制为 49 帧后配一条虚构轨迹用于本训练。此类数据应从本任务中排除。

### 信息不足时的记录格式

建立 `reports/blocked_samples.csv`，至少包含：

```csv
sample_id,source_path,reason,needed_information
```

原因应具体，例如 `unknown_pose_direction`、`unknown_coordinate_system`、`no_intrinsics`、`camera_video_time_mapping_missing`。

---

## 3. 输出目录约定

推荐在**新的输出目录**创建以下结构。不要覆盖原始数据。

```text
ucpe_dataset/
├── videos/
│   ├── 000001.mp4
│   └── 000002.mp4
├── cameras/
│   ├── 000001.json
│   └── 000002.json
├── metadata.csv
├── dataset_manifest.csv
└── reports/
    ├── preprocessing_summary.json
    ├── rejected_samples.csv
    ├── blocked_samples.csv
    └── checks/
```

`metadata.csv` 中的相对路径以 `ucpe_dataset/` 为根目录。路径使用 `/`，不要写入原机器的绝对路径。

建议把原始样本 ID、源文件路径、变换规则和输出 ID 写入 `dataset_manifest.csv`，保证以后可追溯：

```csv
sample_id,source_video,source_camera,video,camera_trajectory,frame_count,status
000001,/original/example.mp4,/original/poses.json,videos/000001.mp4,cameras/000001.json,49,accepted
```

---

## 4. 视频处理规则

### 4.1 推荐先统一为 49 帧片段

当前启动脚本默认 `num_frames=49`。为减少短视频被静默缩短的歧义，推荐离线裁剪为**恰好 49 个解码帧**。

- 保留原视频时间顺序，不随机打乱帧；
- 先按连续、无场景切换的窗口切段；
- 不足 49 帧的输入片段默认跳过，**不要循环播放、复制尾帧或倒放补帧**；
- 长视频可以生成多个不重叠或可控重叠的窗口；在 manifest 记录源片段起止帧或时间戳；
- 相邻窗口重叠率要记录，避免训练/验证集出现重复内容；
- 按源视频划分 train/validation，不能按窗口随机拆分同一源视频。

若要保留更长视频，训练 loader 会读取**前 `num_frames` 个解码帧**，所以相机文件也必须以这段前缀为基准。少于 49 帧的视频可能被 loader 缩短到满足 `4N+1` 的长度；本数据准备任务应通过输出恰好 49 帧来避免这种隐式行为。

如果将来改用 121 帧，必须同时修改训练脚本、重新输出或检查相机文件，并确认它覆盖 `0..120`。不能仅靠重复 49 帧相机位姿扩展到 121 帧。

### 4.2 帧率、时间戳和重采样

当前训练入口默认 `fix_frame_rate=False`：按解码顺序读取视频帧，**不会根据相机时间戳自动重采样视频**。

因此有两种合法策略，优先使用第一种：

1. **保持原始视频帧序。** 相机按原始帧号关联。输出 49 帧窗口和完全对应的 49 个相机条目。
2. **显式重采样。** 如果要统一到指定 FPS，必须依据时间戳同时重采样视频和相机轨迹，重命名输出帧索引为 `0..48`，并保存映射。不能只改视频 FPS 或只插值位姿。

对相机重采样时：

- 时间对应应使用原始时间戳，而不是假装 RealEstate10K 的微秒时间戳是帧号；
- 只在内插两端都有可靠样本时插值，避免长时间外推；
- 平移可线性插值；旋转应使用四元数 SLERP 或旋转李代数插值，不要直接对矩阵元素插值后当作旋转；
- 内参变化时也要同步处理；
- 把 `source_frame_index`、源时间戳、输出帧索引映射写入 sidecar 或 manifest。

不要默认把相机“等间隔”就等同于视频“等间隔”。丢帧、可变帧率视频必须按时间戳处理。

### 4.3 分辨率、裁剪与内参（容易出错）

**推荐输出已经 cover-resize + center-crop 到 480×832 的视频，并把内参同步转换到输出图像坐标。** 然后相机 JSON 使用：

```json
"image_size": [832, 480],
"intrinsics_already_cropped": true
```

这里的 `image_size` 是 `[width, height]`，不是 `[height, width]`。此模式下训练 loader 不再对 K 做第二次裁剪。

设源图像宽高为 `Ws, Hs`，目标为 `Wt=832, Ht=480`：

```text
scale = max(Wt / Ws, Ht / Hs)
resized_width  = round(Ws * scale)
resized_height = round(Hs * scale)
crop_left = round((resized_width  - Wt) / 2)
crop_top  = round((resized_height - Ht) / 2)

fx' = fx * scale
fy' = fy * scale
cx' = cx * scale - crop_left
cy' = cy * scale - crop_top
```

**实现注意：** 与 DiffSynth 的 `ImageCropAndResize` 保持一致时，Python 的 `round` 采用 bankers rounding。例如 `round(10.5) == 10`，不是 11。不要混用 `floor(x+0.5)`。如果使用其他语言或图像库，必须用样例验证最终 crop offset 一致。

对每帧内参分别执行上述操作。不要把 `intrinsics` 的顺序误认为 `[cx, cy, fx, fy]`；正确顺序是 `[fx, fy, cx, cy]`。

**不要：**

- 直接 stretch 到目标分辨率却仍套用正方形像素 center-crop 公式；
- 裁剪视频但忘记移动 `cx/cy`；
- 缩放视频但保留原始焦距；
- 只缩放 `fx/fy`、忘记 `cx/cy` 也随图像缩放；
- 把焦距乘 `scale` 后再次传成“原始尺寸内参”；
- 对已设为 `intrinsics_already_cropped=true` 的 K 再次裁剪。

如果输出保留原始尺寸，可以不提前 resize/crop，但必须把源尺寸和源像素 K 写入 JSON，并省略或设 `intrinsics_already_cropped=false`。训练时 DiffSynth 会对视频和 K 做同样的 cover-resize + center-crop。**同一数据集不要混用未声明的两种 K 语义。**

### 4.4 畸变、鱼眼和特殊相机

UCPE 支持 `x_fov + xi` 的 UCM 表示，但发生宽高比改变的 center crop 时，本实现会拒绝直接裁 UCM 参数。不要伪造普通针孔 K 来绕过错误。

可选处理：

- 先用已标定畸变模型将图像去畸变，并同时计算新 K，再执行上述裁剪；或
- 保持 UCM 原始表示，不做宽高比改变的 crop；或
- 将该样本隔离并记录，等待针对鱼眼模型的专用转换。

---

## 5. 相机 JSON 输出契约

每个样本一个 JSON。推荐固定使用下面的**每帧显式内参**格式，避免 zoom 或动态标定被静默丢弃：

```json
{
  "format_version": 1,
  "coordinate_system": "opencv",
  "image_size": [832, 480],
  "intrinsics_already_cropped": true,
  "intrinsics_normalized": false,
  "translation_scale": 1.0,
  "frames": [
    {
      "frame_index": 0,
      "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "intrinsics": [600.0, 600.0, 416.0, 240.0]
    }
  ]
}
```

上面的矩阵仅用于展示 JSON 结构，不是可复制的真实轨迹。实际 `frames` 通常有 49 项，`frame_index` 为 `0..48`。每项都需有 4×4 `c2w` 和 `[fx,fy,cx,cy]`。

`format_version` 是当前数据集的附加追溯字段，加载器会忽略它。不要增加与加载器冲突的顶层 `c2w`、`w2c`、`intrinsics` 或 `frame_indices` 字段；使用 `frames` 时位姿和内参放在各帧对象内。

### 坐标系约定

OpenCV 相机坐标：**x 向右、y 向下、z 向前**。列向量几何关系为：

```text
p_camera = R_w2c * p_world + t_w2c
p_world  = R_c2w * p_camera + t_c2w
c2w = inverse(w2c)
```

若原始位姿是 OpenGL/NeRF（x 右、y 上、z 后）：

```text
c2w_opencv = c2w_opengl @ diag(1, -1, -1, 1)
```

转换后应统一输出 `coordinate_system="opencv"`，避免下游二次转换。

**重要：** 不要把 `w2c` 的矩阵直接改名为 `c2w`；求逆才是正确变换。不要把相机中心直接放到 `w2c` 的最后一列。不要在不知道约定时通过“试一下另一个方向”随意选择；应通过投影、可视化或源数据集规范验证。

### 位姿的合法性

逐帧检查：

- 形状为 4×4，最后一行为 `[0,0,0,1]`（允许数值误差）；
- 旋转块近似正交，行列式接近 +1；
- 不含 reflection、非均匀 scale 或非法投影项；
- `fx>0`、`fy>0`，全部数值有限；
- `c2w[:3,3]` 是**世界坐标中的相机中心**，不是沿相机局部轴的位移；
- 不使用大量完全重复位姿掩盖跟踪丢失。

默认训练会把所有位姿变换为**首帧相机相对坐标**，所以你不必在 JSON 中自行再相对化。若已经相对化也可以，但不得重复执行两种归一化。推理轨迹必须遵循相同语义。

### 平移尺度

真实世界米制平移可作为基线，但 UCPE 的 ray 对尺度敏感。SfM 通常只有任意尺度，不能声称它是米制。

选择并记录一种一致方案：

- 可靠米制/深度尺度：`translation_scale=1.0`；
- 每段按可靠 near-depth 归一化：`translation_scale=1/near_depth`；
- 其他方案：必须先做小规模实验，记录统计和可视化，不可事后每个样本随意缩放来“让轨迹看起来更大”。

`translation_scale` 必须为正，训练和推理保持一致。不要把未知尺度写成 1 米，也不要把姿态旋转矩阵乘上平移尺度。

---

## 6. Prompt 处理

优先使用已有、与片段内容对应的人工标注。若用视觉模型自动描述，prompt 应描述可见主体、动作、场景、光照和镜头信息，避免不可见事实和臆测。

- 每行 prompt 对应输出视频本身，而不是长视频的另一个片段；
- 不要写“相机向右”来替代真实位姿标签；
- 可保留真实可见的镜头运动描述，但它不是相机 JSON 的替代品；
- 去除个人信息、乱码和无意义模板重复；
- `metadata.csv` 必须用标准 CSV 引号转义，不要用逗号拼接字符串。

示意（不是可复制的内容标签）：

```csv
video,prompt,camera_trajectory
videos/000001.mp4,"A person walks through a sunlit corridor.",cameras/000001.json
```

---

## 7. 实际转换程序应如何组织

将处理程序放在**单独的脚本目录**，不要放进原始数据目录或下载依赖目录，避免同名模块污染导入。程序应具备：

1. `discover`：发现输入及格式，不修改数据；
2. `inspect`：解码样例，读取尺寸、帧数、FPS、时间戳、K、位姿约定；
3. `convert`：显式执行坐标变换、视频/相机同步和图像/K 同步；
4. `validate`：独立重新读取产物，不只检查内存对象；
5. `report`：输出统计、失败原因和人工复核清单。

核心转换可以使用下面的伪代码，但必须用真实格式适配：

```python
for sample in discovered_samples:
    video_frames, timestamps = decode_in_display_order(sample.video)
    poses, intrinsics = read_real_calibration(sample.camera)
    verify_coordinate_system_and_pose_direction(sample.metadata)

    windows = split_at_scene_changes_and_valid_tracking(video_frames, poses)
    for window in windows:
        window = select_exactly_49_frames_without_padding(window)
        video = crop_resize_video(window.frames, target=(480, 832))
        K = crop_resize_intrinsics(window.intrinsics, source_size, target=(480, 832))
        c2w = convert_to_opencv_c2w(window.poses)
        assert temporal_alignment_is_verified(video, c2w, window.timestamps)
        write_video_and_camera_json(video, K, c2w)
        append_metadata_and_manifest()
```

不要复制伪代码后直接报告成功。必须完成实际格式适配、实际解码、实际文件写入及错误处理。

建议批处理程序默认不覆盖输出；如需重跑，使用明确的新目录或让用户确认覆盖。失败的样本继续处理并记录，不要因单个坏文件丢掉整批进度。

---

## 8. 交付前必须执行的校验

### 8.1 结构校验

- `metadata.csv` 可被标准 CSV 解析，表头恰为 `video,prompt,camera_trajectory`；
- 每行路径存在、无重复 sample ID、无空 prompt；
- 每个视频能被完整解码，帧数与 JSON 帧数一致；
- 输出帧数为 49，索引严格为 `0..48`；
- JSON 为 UTF-8，所有位姿和内参为有限数值；
- 视频像素尺寸与 JSON `image_size` 一致；
- 本数据集统一使用正确的 `intrinsics_already_cropped`；
- manifest 的成功数量与 metadata 行数一致。

### 8.2 几何与时间校验

- 投影或可视化若干源点，检查转换前后落点一致；
- 对静态背景占主导的场景，检查位姿运动和相邻帧光流的大方向是否合理；
- 检查连续相机中心没有异常跳变，旋转没有突然 180° 翻转；
- 检查首帧、中间帧、末帧的裁剪区域、K 和可视化 ray 一致；
- 检查重采样样本的源帧/时间戳映射，不以文件行数相同代替同步验证；
- 抽看至少每类输入格式的若干样本，以及所有自动修复样本。

### 8.3 使用训练仓库加载器做最终校验

在安装了 DiffSynth 依赖的环境中，**逐个**读取产物：

```python
from pathlib import Path
import csv
from diffsynth.core.data.operators import LoadVideo, ImageCropAndResize, ToAbsolutePath
from diffsynth.models.wan_video_ucpe import (
    LoadUCPECameraTrajectory,
    build_ucpe_ray_features,
)

root = Path("/absolute/path/to/ucpe_dataset")
load_camera = ToAbsolutePath(str(root)) >> LoadUCPECameraTrajectory()

with (root / "metadata.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        video = LoadVideo(
            num_frames=49,
            time_division_factor=4,
            time_division_remainder=1,
            frame_processor=ImageCropAndResize(height=480, width=832),
        )(str(root / row["video"]))
        assert len(video) == 49, (row["video"], len(video))
        assert video[0].size == (832, 480)

        trajectory = load_camera(row["camera_trajectory"])
        rays = build_ucpe_ray_features(
            trajectory=trajectory,
            num_frames=49,
            height=480,
            width=832,
            latent_frames=13,
            vae_stride=(4, 16, 16),
            patch_size=(1, 2, 2),
            condition="relray_absmap",
        )
        assert rays.shape == (1, 13, 15, 26, 19)
```

这段检查只证明格式、几何构造和 ray 网格可被加载，**不能证明相机方向、时间同步、数据质量或训练效果正确**。它仍需要前面的可视化和人工/自动质量检查配合。

若 ray 构造报 `missing sampled RGB frames`，通常是相机文件不足、帧号不对应或视频被缩短，不要通过复制最后一个位姿修复。

---

## 9. 推荐报告格式

`reports/preprocessing_summary.json` 至少包含：

```json
{
  "target": {"height": 480, "width": 832, "num_frames": 49},
  "source_format_summary": {},
  "accepted_samples": 0,
  "rejected_samples": 0,
  "blocked_samples": 0,
  "coordinate_system_output": "opencv",
  "intrinsics_mode": "cropped_480x832_pixels",
  "translation_scale_policy": "describe actual policy",
  "frame_resampling_policy": "describe actual policy",
  "geometry_visual_checks": {"checked": 0, "passed": 0, "failed": 0},
  "loader_checks": {"checked": 0, "passed": 0, "failed": 0},
  "unresolved_questions": []
}
```

不要在没有运行检查时把数量写成通过，也不要把“脚本执行结束”描述为“全部样本校验通过”。报告失败日志和具体样本路径。

---

## 10. 最终向用户交付的说明

完成后用简洁报告回答：

1. 原始输入是什么格式，发现了多少视频/片段；
2. 成功转换多少、跳过多少、阻塞多少；
3. 输出数据集的绝对路径；
4. 视频尺寸、帧数、坐标系、K 的含义和平移尺度策略；
5. 实际执行了哪些检查，哪些只是抽样检查；
6. 哪些样本需要人工确认；
7. 训练启动命令：

```bash
DATASET_PATH=/absolute/path/to/ucpe_dataset \
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-UCPE.sh
```

如果原始数据只有无标定视频，明确说明还需要相机估计，不能声称已完成可直接训练的 UCPE 数据集。

相关实现说明：`examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-UCPE.md`。
