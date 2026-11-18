# 静态场景训练数据处理、坐标系与验收规范

本文面向准备 Uni3C 静态场景训练数据的人或其他 AI。目标不是让画面所有像素不变，而是：**只改变相机，人物姿态、表情、物体状态和背景内容不自主变化，同时保留相机引起的透视、视差、遮挡和去遮挡。**

请先区分两条流程：

1. **虚拟轨迹生成/推理**：`cam_render.py` 根据单张参考图、单目深度和指定参数生成相机轨迹及点云条件。这是当前完整支持的入口。
2. **真实视频训练监督**：`target_video` 必须来自真实静态场景的新视角，且逐帧相机与 render 一致。当前仓库**没有**从任意 `target_video` 自动恢复相机并导出 `cam_info.json` 的完整入口。未标定视频必须先做 SfM/SLAM/采集标定，再编写转换和渲染脚本复用 `point_rendering()`。

训练、缓存、910B2 命令见 [910B2 静态场景训练指南](ascend_static_scene.md)；主体中心运镜参数见 [主体中心运镜文档](subject_centered_camera.md)。

---

## 1. 最重要的数据契约

每个训练样本由以下五部分组成：

| 文件 | 含义 | 不能误解为 |
|---|---|---|
| `reference_image` | 目标轨迹第 0 帧对应的 RGB 参考图，供深度、CLIP 和 Wan I2V 首帧条件使用 | 任意一张同场景照片 |
| `target_video` | 同一静止场景沿目标真实相机轨迹采集的逐帧 RGB 监督 | Uni3C 生成结果、render 的副本、带人物动作的视频 |
| `render.mp4` | 用参考图单目点云，按照 `cam_info.json` 中同一组相机投影得到的几何条件 | 真实目标视频、完整三维重建 |
| `render_mask.mp4` | 点云投影的缺失区域 mask；白表示没有有效投影点，黑表示有投影点 | 人物/前景分割 mask |
| `cam_info.json` | 与 render 逐帧一致的 `w2c`、像素内参和 render 尺寸 | `c2w`、归一化内参、OpenGL 外参 |

逐样本必须满足：

```text
reference_image == target_video 的第 0 帧所对应的同一次成像
render[i]         == 参考点云经 cam_info 中第 i 个相机投影的结果
mask[i]           == render[i] 的缺失区域
target_video[i]   == 第 i 个真实相机视角下的静止场景监督
```

其中 `==` 表示几何和帧顺序对应，不要求 render 与真实照片像素相同。单目点云无法恢复真实背面、复杂材质和所有遮挡，mask 中的洞正是需要扩散模型补全的区域。

### 1.1 绝对不能这样构造训练集

- 给任意真实环绕视频随手指定 `--d_phi`，再把生成的 render 当作对齐条件。
- 把 `render.mp4` 或现有 Uni3C 输出复制成 `target_video`。
- 使用人物走路、说话、做手势的视频，只在 prompt 中写“保持静止”。
- 把每个目标帧都监督回参考图，以消除所有像素运动。
- 将 `cam_info.json["extrinsic"]` 写成 `c2w`，或预先翻转 y/z 来“适配 OpenCV”。
- 裁切/电子防抖改变图像，却不更新内参和逐帧对应关系。
- 只看第 0 帧 render 判断对齐；代码会强制替换首帧，首帧看起来正确不能证明轨迹正确。

### 1.2 当前脚本会检查与不会检查的内容

`src/dataset.py` 和缓存脚本会检查：

- target/render/mask 的实际帧数是否都等于 `--nframe`；
- `cam_info.json` 的 pose 数是否等于帧数；
- `cam_info.json` 的 `width/height` 是否等于 render 首帧实际尺寸；
- 存在 `per_frame_intrinsic` 时，其数量是否等于帧数；
- 缓存张量的通道、latent 帧数和空间形状。

它们**不会自动证明**：

- reference 真的是 target 第 0 帧；
- target 中的人和物体真的静止；
- 外参方向、相机轴或尺度正确；
- target 与 render 使用同一轨迹；
- mask 文件尺寸、语义和视频压缩是否正确；
- 图像裁切、EXIF、镜头畸变和电子防抖是否处理正确。

因此必须执行本文第 8 节的人工与几何验收。

---

## 2. 推荐采集和筛选方法

### 2.1 优先级

推荐顺序：

1. **已知标定的滑轨、机械臂或运动控制系统**：直接记录逐帧相机位姿，最可靠。
2. **同步多相机阵列**：场景和人物保持同一瞬间，各相机天然同步；按一条明确空间路径排列视角，并保留每个相机的 K 和外参。
3. **静态场景视频 + 高质量 SfM/SLAM**：只使用背景静止、纹理充分、估计稳定的片段；动态区域应在重建时 masking，并对齐结果做人工检查。
4. 普通未标定人物视频：不推荐。没有可靠相机时不能用于本任务的对齐监督。

同步多相机输出不一定按采集时间形成视频，但可以按已知空间路径排序成“相机视频”。应检查路径是否连续、视角是否适合 16 fps 的平滑运镜，并尽量统一曝光、白平衡和色彩。

### 2.2 场景静止性

保留：

- 静止家具、建筑、雕塑、摆件；
- 固定姿态的人体模型或确实能保持静止的主体；
- 固定照明下，仅相机运动导致的视差、透视和遮挡变化；
- 合理的视角相关反射、高光变化。

剔除或单独标记：

- 走动、手势、头部转动、说话、明显眨眼或姿态漂移；
- 明显呼吸、衣物飘动、头发摆动；
- 树木摆动、水面、行人、车辆、屏幕内容变化；
- 自动曝光、自动白平衡、灯光闪烁、移动阴影；
- 镜头切换、转场、重复帧、大运动模糊和严重滚动快门；
- 自动对焦或数码变焦造成、但没有逐帧标定的尺度变化。

“背景光流不为零”不代表场景在运动，因为相机本身会产生光流。不要通过简单光流阈值或相邻帧 SSIM 自动删除合法视差。应在相机补偿后检查残差，并结合逐帧人工检查。

### 2.3 拍摄建议

- 锁定曝光、白平衡、对焦和焦距；优先全局快门或较短曝光。
- 关闭电子增稳、自动构图和未知后期 warp；若无法关闭，必须恢复逐帧等效 K/位姿。
- 使用固定光照，避免镜面占主导和纯色无纹理背景。
- 覆盖环绕、俯仰、推拉、平移、变焦及组合轨迹，同时覆盖人物、物体、室内、室外、近远景。
- 训练/验证按**场景**划分，不要把同一场景的不同片段同时放进训练集和验证集。

数据量没有通用保证。可以先用少量高质量样本跑通，再扩展到数百至数千片段；对齐质量优先于数量。

---

## 3. 图像、视频、帧和尺寸处理

### 3.1 先固定最终几何预处理

推荐顺序：

```text
原始素材
  -> EXIF 方向烘焙
  -> 去畸变（若需要）
  -> 选定统一 crop/resize
  -> 同步更新 K、subject 像素和深度图
  -> 选择连续 N 帧
  -> 导出 target/reference
  -> 根据相同 N 帧相机渲染 render/mask
  -> 验收后生成 JSONL 和缓存
```

不要在完成标定后又改变 crop、旋转或长宽比而不更新 K。

**建议把 reference 交付为 EXIF 已烘焙的 RGB PNG。** `cam_render.py` 会执行 `ImageOps.exif_transpose()`，但 `src/dataset.py` 直接 `Image.open()` 后 resize；依赖 EXIF 的软件与直接解码器可能得到不同方向。最稳妥的方法是提前把方向写进像素，再清除方向标签。

### 3.2 帧数和顺序

- 先使用 `N=17` 验证，再考虑 33/49/81。
- Wan 时序 VAE 要求 `N % 4 == 1`。
- `target_video` 必须包含第 0 帧，而不是从第 1 帧开始。
- target/render/mask/`extrinsic`/`per_frame_intrinsic` 均为一一对应的 N 项。
- 当前读取器按解码顺序取帧，不使用时间戳自动同步；FPS 主要是导出播放参数。
- 不要用光流补帧、重复帧或随意 fps 转换制造 N 帧。若重采样时间，必须同步重采样/重标定相机，且避免把插值伪影当监督。

如果已经把最终帧导出为 `frames/000000.png` 起的连续序列，可这样编码：

```bash
ffmpeg -framerate 16 -start_number 0 \
  -i frames/%06d.png -frames:v 17 \
  -c:v libx264 -crf 14 -pix_fmt yuv420p target.mp4
```

从**最终 target** 解码第 0 帧可排除 reference/target 首帧预处理不一致：

```bash
ffmpeg -i target.mp4 -frames:v 1 reference.png
```

若希望 reference 保留未压缩原图，也必须保证它与最终 target 第 0 帧来自同一帧、同一视角和相同几何预处理；少量编码差异可以接受，明显曝光、裁切或内容差异不可接受。

检查真实解码帧数，而不是只信容器元数据：

```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames,r_frame_rate,avg_frame_rate \
  -of json target.mp4
```

### 3.3 两级 resize 不要混淆

**渲染阶段 `cam_render.py`：**

- 从五组 `(H,W)` 中选择与参考图宽高比最接近的一组：`(480,768)`、`(512,720)`、`(608,608)`、`(720,512)`、`(768,480)`；
- 只 resize，不 crop；
- 深度和 K 在这个 render 像素网格上产生；
- `cam_info.json.width/height` 记录这个输出尺寸。

**训练/推理读取 `src/dataset.py`：**

- 再根据参考图宽高比、`max_area` 和 16 的倍数约束计算模型输入 `(H_model,W_model)`；
- reference/render 使用 bicubic resize，mask 使用 nearest resize并阈值化；
- target 在缓存脚本中 resize 到同一个模型尺寸；
- K 的行按尺寸比例缩放，见第 5.4 节。

所以：

- `cam_info.width/height` 是**磁盘 render 的尺寸**，不一定等于最终模型尺寸；
- target、reference、render 最好具有相同宽高比，避免各自独立 resize 造成几何差异；
- 若改变 `max_area`，通常不必重写原始 K，读取器会缩放；但必须重新生成缓存；
- 默认 batch size 为 1。若使用更大 batch，相同样本形状还需满足 DataLoader 的拼接要求。

### 3.4 镜头畸变和裁切

当前相机模型是无畸变针孔模型，只接受 K，没有径向/切向畸变参数。

- 鱼眼、广角或明显桶形畸变素材必须先去畸变，再使用去畸变后的 K 和图像。
- COLMAP 的 `SIMPLE_PINHOLE`/`PINHOLE` 可读取针孔参数；带畸变模型不能忽略畸变系数。
- crop 左上角为 `(left,top)` 时，在以像素中心坐标表示的 K 中：
  `cx' = cx-left, cy' = cy-top`，焦距不变。
- 随后从 `(W,H)` resize 到 `(W',H')`，令 `sx=W'/W, sy=H'/H`：
  `fx'=sx*fx, cx'=sx*cx, fy'=sy*fy, cy'=sy*cy`。
- 深度是相机 z，不因图像 resize 而乘 `sx` 或 `sy`。

---

## 4. 当前仓库的坐标系

### 4.1 相机局部坐标：OpenCV 风格

```text
+x：图像右方
+y：图像下方
+z：相机前方，可见点通常 Z>0
```

深度是正的 **camera-z**，不是像素到相机的欧氏距离，也不是 disparity。

对第 i 帧，`cam_info.json["extrinsic"][i]` 是世界到相机矩阵 `w2c_i`：

```text
w2c_i = [ R_i  t_i ]
        [ 0    1  ]

P_camera = R_i * P_world + t_i
P_camera_homogeneous = w2c_i * P_world_homogeneous
c2w_i = inverse(w2c_i)
```

`c2w_i[:3,3]` 是相机中心在世界坐标中的位置；`c2w_i[:3,:3]` 的三列是相机 x/y/z 轴在世界坐标中的方向。

`src/dataset.py` 将 JSON 的 `extrinsic` 直接命名为 `w2cs`；`src/camera.py` 再求逆得到 `c2w`。不要根据其他项目里 `extrinsic` 的习惯猜方向。

### 4.2 像素坐标与针孔投影

图像数组约定：

- `u` 向右，`v` 向下；
- 左上角像素索引为 `(0,0)`；
- 像素中心为 `(u+0.5,v+0.5)`；
- 坐标写成 `(u,v)`，不是 `(row,column)`，也不是 `(x,y)` 的归一化坐标。

点云反投影和主体提升使用：

```text
P_camera = Z * inverse(K) * [u+0.5, v+0.5, 1]^T
P_world  = c2w[:3,:3] * P_camera + c2w[:3,3]
```

等价投影为：

```text
u_center = fx * X/Z + cx
v_center = fy * Y/Z + cy
```

K 使用**当前图像像素单位**：

```text
K = [ fx   0  cx ]
    [  0  fy  cy ]
    [  0   0   1 ]
```

`fx/fy` 不是 0～1 归一化焦距，`cx/cy` 也不是 0～1 坐标。当前代码预期零 skew、方形像素不是强制条件，但必须使用真实标定的 `fx/fy`。

### 4.3 `cam_render.py` 默认构造的世界坐标

外部接口的相机局部坐标是 OpenCV 风格，但**世界坐标不是固定的重力/地图坐标系**。默认世界由参考图深度和初始相机构造：

```python
c2w_0 = Rx(-start_elevation) @ Translate(0, 0, -radius)
w2c_0 = inverse(c2w_0)
```

其中：

- `radius` 是 CarveKit 前景 mask 内原始预测深度的中位数；
- 当仰角为 0 时，首相机位于 `(0,0,-radius)`，朝世界 `+z` 看，世界 `+x` 向右、`+y` 向下；
- 默认主体/前景中心深度面位于世界原点附近，因此预设轨迹默认围绕世界原点运动；
- 非零 `start_elevation` 会旋转首相机，不应再简单声称世界 y 就是真实重力方向；
- 世界平移、旋转和均匀尺度在数学上可整体变化，只要参考点云和所有相机使用同一个世界。

`point_rendering()` 内部会将 OpenCV 相机局部 x/y 轴适配到 PyTorch3D 渲染约定。**这是内部实现，不要求在 JSON 中预先保存 OpenGL 外参；预先翻转会破坏接口。**

如果外部标定数据直接调用 `point_rendering()`，可以保留经过统一相似变换后的外部世界，不必强行使用默认世界原点。关键是：

```text
参考点云 P_world = c2w_0 * (Z * K_0^-1 * pixel)
所有 target 相机 w2c_i 必须作用于同一个 P_world
```

### 4.4 世界单位与尺度

MoGe/Depth-Pro 输出的 z 通常被称为 metric depth，但单目预测的绝对尺度仍可能有误差。本仓库的点云、首相机半径、轨迹平移和主体中心都共享该深度单位。

要求：

- `w2c/c2w` 的平移和参考点云深度使用同一尺度；
- 旋转矩阵无量纲，尺度只作用于三维点、相机平移和深度；
- SfM 的任意尺度必须与单目深度通过已知基线或稳健相似变换对齐；
- 不能只对点云乘尺度而保留旧外参；
- ray embedding 后续还有自身归一化，但这不会修复 render 中错误的相机/深度尺度。

### 4.5 ray embedding 还会做首帧归一化

`get_camera_embedding(..., normalize=True)` 不直接使用原始 w2c 的平移生成 Plücker ray，而是先调用 `camera_center_normalization()`：

1. 用首帧 `c2w_0` 左乘全部 w2c，使归一化世界的首帧旋转为单位阵，即以首相机坐标系表达轨迹；
2. 对筛选后的相机中心集合减去均值中心；
3. 缩放相机平移，使用 `camera_scale=2.0`；
4. 在归一化世界中计算每像素方向 `rays_d`、相机原点 `rays_o`，得到 `[rays_d, cross(rays_o,rays_d)]` 六通道。

所以：

- render 始终使用 `cam_info.json` 的原始 w2c 和深度尺度；
- ray embedding 使用归一化后的轨迹，二者的相机平移数值不能直接比较；
- 原始相机平移和点云深度仍必须先处于同一物理尺度，ray 的后处理不能修复 render 的尺度错误；
- 训练和推理共用该归一化路径，数据处理者不要预先对 JSON 再做一次 ray 专用归一化。

---

## 5. `cam_info.json` 完整规范

### 5.1 字段

| 字段 | 必需性 | 形状/单位 | 含义 |
|---|---|---|---|
| `width`, `height` | 必需 | 整数像素 | 磁盘 render 帧尺寸，必须与实际文件一致 |
| `intrinsic` | 必需 | `[3,3]`，像素 | 原始/首帧 K；即使有逐帧 K 也仍需保留 |
| `extrinsic` | 必需 | `[N,4,4]` | 每帧 **w2c**，不是 c2w |
| `per_frame_intrinsic` | 可选 | `[N,3,3]`，像素 | 变焦或多相机样本的每帧实际 K；存在时优先使用 |
| `depth_model` | 可选元数据 | 对象 | 记录深度后端，不影响 ray 数学 |
| `subject` | 可选元数据 | 对象 | 主体点击、深度和世界点等信息 |

单帧 schema 示例如下。它只用于展示合法 JSON 类型；真实运镜训练通常使用 17 帧，列表必须展开为 N 个完整矩阵：

```json
{
  "width": 768,
  "height": 480,
  "intrinsic": [
    [600.0, 0.0, 384.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0]
  ],
  "per_frame_intrinsic": [
    [
      [600.0, 0.0, 384.0],
      [0.0, 600.0, 240.0],
      [0.0, 0.0, 1.0]
    ]
  ],
  "extrinsic": [
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]
  ]
}
```

固定 K 的样本可以省略 `per_frame_intrinsic`，读取器会把 `intrinsic` 复制 N 份。变焦时不要只写末帧 K 或自己假设插值公式，应保存实际用于渲染的每帧 K。当前 `focal_length` 渐变和 offset 的末帧系数并不都简单地等于终值参数，因此以导出的逐帧矩阵为准。

### 5.2 数值验收

对每个样本至少验证：

```text
extrinsic.shape == (N,4,4)
K.shape == (3,3)，或 per_frame_intrinsic.shape == (N,3,3)
所有值 finite
每帧最后一行为 [0,0,0,1]
R.T @ R ≈ I，det(R) ≈ +1
inverse(w2c_i) @ w2c_i ≈ I
fx>0, fy>0
width/height 等于 render 实际尺寸
```

随后检查参考点云在 target 各相机中的投影：可见点应满足 `Z>0`、投影范围合理，静态地标在 target/render 中位置一致。只检查矩阵可逆远远不够，转置的 R 或写反的 w2c/c2w 仍可能通过数值检查。

### 5.3 从常见外部格式转换

#### 已经是 OpenCV 相机坐标

若外部提供 `w2c_ext` 和相同世界中的三维点，只需要统一世界相似变换，不要翻转相机轴。

若外部提供 `c2w_ext`，则：

```text
w2c_ext = inverse(c2w_ext)
```

COLMAP `images.txt` 的 `qvec,tvec` 通常描述 world-to-camera：`qvec=(qw,qx,qy,qz)` 转为 R，再拼 `[R|t]`。COLMAP 相机局部坐标也是 x 右、y 下、z 前；但其世界尺度和方向是任意的。使用去畸变后的图像、相机和点云，保持帧名映射，不要按文件名字典序猜测时间顺序。

#### 外部是 OpenGL 相机局部坐标

OpenGL 常见约定为 x 右、y 上、z 向后。令：

```text
D = diag(1,-1,-1)
P_opencv = D * P_opengl
D^-1 = D
```

在同一个外部世界中：

```text
w2c_opencv = D * w2c_opengl
c2w_opencv = c2w_opengl * D
```

不要把 D 再错误地乘到世界点上。若世界轴也要改变，应另用下面的世界相似变换。

#### 世界旋转、平移和尺度统一

设外部世界点为 `P_ext`，目标 Uni3C 世界点为：

```text
P_uni = s * Q * P_ext + q
S  = [ sQ  q ]       S^-1 = [ Q^T/s  -Q^T*q/s ]
     [ 0   1 ]              [   0          1   ]
```

其中 Q 为旋转，s>0。则：

```text
c2w_uni_i = S * c2w_ext_i
w2c_uni_i = w2c_ext_i * S^-1
```

K 不因世界相似变换而改变；同一物理点的 camera-z 和外部深度会乘 s。可以变换相机来适配固定单目点云，也可以变换点云和深度来适配真实世界，但二者只能作为一个整体处理。

#### 归一化 K 或 FOV 转像素 K

若归一化约定明确为相对宽/高：

```text
fx_px = fx_norm * W     cx_px = cx_norm * W
fy_px = fy_norm * H     cy_px = cy_norm * H
fx_px = W / (2*tan(fov_x/2))
fy_px = H / (2*tan(fov_y/2))
```

不同 SLAM/AR 框架可能采用不同归一化定义，必须先查其文档并用一个已知三维点投影验证，不能盲用公式。

### 5.4 模型输入 resize 时 K 如何处理

`src/dataset.py` 从 render 尺寸 `(W0,H0)` 到模型尺寸 `(W,H)` 执行：

```python
K[0, :] *= W / W0
K[1, :] *= H / H0
```

逐帧 K 同理。相机外参不会因图像 resize 改变，深度也不会乘图像缩放系数。

不要在 JSON 中同时保存“已经缩放到模型尺寸的 K”和“原 render 的 width/height”，否则会缩放两次。

---

## 6. 深度、点云和 mask 的注意事项

### 6.1 深度来源与有效性

当前 `cam_render.py` 默认使用 MoGe-3 ViT-L，也可选择其他已实现后端。深度必须与 resize 后的 reference RGB 逐像素对齐，输出正 camera-z。

- 无效/非正深度会被保守填补到有效区域的最大远处深度；填充不代表恢复了真实天空或远处几何。
- 深度边界、细头发、透明物体、镜面和天空容易出错。
- 使用真实标定 K 时，可用标定 K 反投影模型预测的 z；不要同时混用估计 K 和真实 K。
- 预测深度与真实相机平移尺度必须通过可见地标和视差验证。

### 6.2 远深度收缩

点云实际渲染前使用 `contract_depth()`，默认：

```text
D = 8 * median(all_reference_depth)
Z' = Z                                      , Z <= D
Z' = 2D - D^2/(Z+1e-6)                      , Z > D
```

远处深度渐近压缩到 `2D`。因此：

- 近处保留原 z，远处不再按真实尺度无限延伸；
- 主体中心模式也应用相同收缩，保证轨道中心和 render 几何一致；
- 外部标定若强调真实远景尺度，需要专门评估这一变换，不能把它当作严格三维重建。

### 6.3 mask 语义和编码

`point_rendering()` 中：

```text
render_mask = 1（视频通常显示白色）：z buffer 没有有效点，需要补全
render_mask = 0（黑色）：至少投影到有效点
```

读取时只取第一个通道，nearest resize 后以 0.5 阈值二值化。白色不是“人物”，黑色也不一定语义上是背景。RGB 空洞也通常为黑色，不能靠 RGB 黑像素代替 mask。

MP4 压缩可能在边缘产生灰值；应检查解码后的黑白比例和边缘，必要时使用高质量/无损导出流程。第 0 帧 render 被替换为 reference，第 0 帧 mask 被置 0，后续仍可能出现空洞。

### 6.4 `pcd.ply` 只是调试导出

当前 `cam_render.py` 的 `pcd.ply` 在正式点云渲染之前导出，使用整数 `[u,v,1]` 反投影和未收缩深度；正式 `point_rendering()` 使用 `[u+0.5,v+0.5,1]` 和收缩后的深度，还会过滤边界点。

因此 `pcd.ply` **不是正式 renderer 内部点云的逐点精确副本**，不要用它做亚像素投影验收或直接替换正式 render。

### 6.5 当前 ray 横向像素网格的实现差异

点云反投影和主体提升使用 `(u+0.5,v+0.5)`；但当前 `src/camera.py::batch_sample_rays()` 实际写为：

```python
x = torch.arange(image_w).float() - 0.5
y = torch.arange(image_h).float() + 0.5
```

也就是 ray embedding 的横向采样为 `u-0.5`，与上述中心约定相差 1 个像素。本文记录的是**当前代码事实**，不是一套理想化约定。

数据处理者不要通过偷偷修改 `cx` 来补偿这一实现差异；训练和推理目前共享同一 loader。若以后修复该几何实现，应单独做代码回归、重新生成受影响缓存并验证 render/ray 对齐，不能只修部分样本。

---

## 7. 真实轨迹如何与 Uni3C render 对齐

### 7.1 当前入口的能力边界

`cam_render.py` 的流程是：

```text
reference -> 单目深度/K -> 初始相机 -> 参数化虚拟轨迹 -> render/mask/cam_info
```

它不会执行：

```text
target_video -> 自动 SfM/SLAM -> 真实轨迹
```

对于外部真实监督，需新增一个数据预处理脚本，复用深度后端和 `src.pointcloud.point_rendering()`，但**不要调用 `build_cameras()` 替换真实轨迹**。输入应是已处理 reference、对齐深度以及逐帧真实 `K_i/w2c_i`。

### 7.2 推荐转换步骤

1. 将 target 固定为连续 N 帧，并建立 `frame_index -> image_name -> camera_id/pose` 的显式表。
2. 完成 EXIF、去畸变、crop/resize；在最终图像上重新确认 K。
3. 从标定结果判断每个位姿到底是 w2c 还是 c2w，确认相机局部轴；用一个已知世界点和图像投影做单元测试。
4. 将所有相机转换到 OpenCV 局部轴，得到 `K_i,w2c_ext_i`。
5. 选一个统一世界。可以保留外部世界，也可以通过 S 将它对齐到参考点云世界。
6. 用第 0 帧 `K_0,c2w_0` 和 reference 深度反投影点云。所有后续相机都作用于这一套点云。
7. 将同一组 `K_i,w2c_i` 同时用于 `point_rendering()` 和 JSON 导出，避免“渲染用一套、ray 用另一套”。
8. 若有变焦/多相机，写 `per_frame_intrinsic`；固定相机可只写 `intrinsic`。
9. 输出 render/mask 后，执行第 8 节的可视化和投影验收。

伪代码：

```python
# 仅说明数据流，不是现有 CLI。
points_world = backproject(reference_depth_z, K[0], c2w[0])
render, mask = point_rendering(
    K=K, w2cs=w2c, depth=reference_depth_z,
    image=reference_rgb, raster_settings=settings, device=device,
)
# 实际 point_rendering 内部自行完成反投影；这里 points_world 用于说明
# 外部点云/相机必须处于同一世界，不应在调用外再变换一次。
save_cam_info(intrinsic=K[0], per_frame_intrinsic=K, extrinsic=w2c)
```

正式实现时直接遵循 `point_rendering()` 的张量形状和首帧替换逻辑，不要把上面的说明性 `backproject` 当成已有仓库函数。

### 7.3 如何验证“真的是同一轨迹”

至少制作 0、1/4、1/2、3/4、末帧的五联图：

```text
target | render | mask | target/render 半透明叠加 | 地标投影
```

检查：

- 主体大小是否随推拉一致变化；
- 左右环绕方向是否一致，近处和远处视差方向是否一致；
- 俯仰方向、水平线、墙面边缘是否一致；
- 遮挡出现/消失的顺序是否合理；
- 地标是否存在单向系统性漂移；
- 首帧 reference 与 target 是否真正对应，而不是被首帧替换掩盖；
- mask 是否随视角合理扩大，而不是所有帧全白或固定不动。

render 与 target 不要求纹理、光照和遮挡区域完全相同；不能把整帧 SSIM/PSNR 低直接判为相机不对齐。优先比较可见静态地标、轮廓和投影几何。

SfM/COLMAP 的重投影误差、track 长度和匹配可视化也应保留。经验上高质量标定通常需要亚像素级重投影表现，但不能只凭一个阈值接受数据；错误尺度和动态物体仍可能产生看似合理的误差。

---

## 8. 主体中心 `subject_pixel` 的坐标和深度

### 8.1 使用哪个图像坐标系

`--subject_pixel U V` 使用：

- **EXIF 校正后的原始 reference 图像**；
- 原图尺寸 `(W0,H0)` 下的像素索引；
- u 向右、v 向下，左上角为 `(0,0)`；
- 不是 resize 后 render 坐标，不是归一化坐标，也不是 target 其他帧坐标。

映射到 render `(W,H)`：

```text
u_render = (u+0.5) * W/W0 - 0.5
v_render = (v+0.5) * H/H0 - 0.5
```

然后用 `(u_render+0.5,v_render+0.5)` 和 K 反投影。图像只做 resize、不 crop 是该入口的假设；如果交付前已 crop，必须将点击同步转换到交付 reference，并确保 K 也更新。

### 8.2 深度采样

默认将映射点取到邻近像素，并在 `--subject_depth_radius 2` 对应的 5×5 邻域内，对有限正深度取中位数。注意：

- 半径单位是 render 像素，不是原图像素；
- 点击应位于主体内部，避开轮廓、头发边缘和前后景深度跳变；
- 半径 0 为单像素采样；主体较小或边缘异常时可调小并检查；
- 取中位数不是实例分割，不能自动保证选中正确物体。

随后应用与 renderer 相同的远深度收缩，再计算：

```text
P_subject_camera = Z_render * K^-1 * [u_render+0.5,v_render+0.5,1]
P_subject_world  = c2w_0 旋转 * P_subject_camera + c2w_0 平移
```

`--subject_depth Z` 是**收缩后的 render depth** 覆盖值，不是原始传感器深度，也不要乘图像 resize 比例。`subject_info.json`/`cam_info.json` 中重点检查：

- `source_pixel`, `resized_pixel`；
- `source_camera_z`/`source_depth`：原始深度采样；
- `render_depth`：收缩后、实际用于轨道中心的深度；
- `camera_point`, `world_point`；
- `initial_c2w`, `world_up`。

### 8.3 居中轨迹不等于全程立即居中

默认世界 up 沿用首相机上方向；`--subject_keep_upright` 才使用 `--subject_world_up`，其默认值为世界 `(0,-1,0)`。这个 up 是渲染世界参考，不保证是真实重力。

第 0 帧保留原始朝向。默认前 10% 轨迹通过 SLERP 平滑转向主体，转向完成后，在无额外平移的组合轨迹中主体投影位于主点。若需与真实轨迹训练对齐，不能先任意生成主体轨迹再把它当作真实相机；仍需真实标定。

---

## 9. 推荐目录和 JSONL

```text
data/static/
├── static_train.jsonl
├── static_val.jsonl
└── samples/
    └── scene000_cam01/
        ├── reference.png
        ├── target.mp4
        ├── render/
        │   ├── render.mp4
        │   ├── render_mask.mp4
        │   ├── cam_info.json
        │   ├── pcd.ply                 # 可选，仅调试
        │   └── subject_info.json       # 可选
        ├── calibration/
        │   ├── frame_camera_map.json   # 帧名、K、原始位姿来源
        │   ├── world_transform.json    # 外部坐标到 Uni3C 的 S/轴转换
        │   └── reconstruction/         # 可选，原始标定结果
        └── qc/
            ├── contact_sheet.jpg
            └── checks.json
```

JSONL 每行一个样本；相对路径以 JSONL 所在目录为基准：

```json
{"sample_id":"scene000_cam01","reference_image":"samples/scene000_cam01/reference.png","render_path":"samples/scene000_cam01/render","target_video":"samples/scene000_cam01/target.mp4","prompt":"A completely static indoor scene; only the camera moves around the motionless person and furniture."}
```

要求：

- `sample_id` 唯一且只能是一个安全文件名组件，不能带目录、`..` 或绝对路径；
- 每个样本使用独立、稳定的 render 目录；
- prompt 描述真实静态内容和相机运动，不用 prompt 掩盖 target 中的动作；
- 额外 JSON 字段不会自动变成几何控制；重要转换应保存到 sidecar 并进入数据处理日志；
- 修改 reference、target、render、mask、cam_info 或 prompt 后必须重新缓存。缓存清单记录路径和 SHA-256，改变输入后应使用新输出目录。

---

## 10. 数据验收清单

### A. 原始素材

- [ ] 场景和人物真正静止，只存在相机导致的变化。
- [ ] 无切镜、补帧、未知防抖、未标定变焦和明显动态光照。
- [ ] 已记录来源、采集/标定方法、镜头模型和许可信息。
- [ ] 训练/验证按场景隔离。

### B. 图像和视频

- [ ] EXIF 已烘焙，RGB 方向一致。
- [ ] 去畸变/crop/resize 已同步更新 K 和 subject 坐标。
- [ ] reference 对应最终 target 第 0 帧。
- [ ] target/render/mask 实际解码均为 N 帧，`N%4==1`。
- [ ] 帧顺序一致，无重复、丢帧和时间错位。
- [ ] 宽高比一致；render 尺寸与 cam_info 一致。
- [ ] mask 白=缺失、黑=有效，解码阈值后合理。

### C. 相机和几何

- [ ] JSON `extrinsic` 是逐帧 w2c，不是 c2w。
- [ ] 相机局部轴为 x 右、y 下、z 前。
- [ ] K 是 render 像素单位，未提前重复缩放。
- [ ] 旋转、末行、逆矩阵、正值深度和有限数检查通过。
- [ ] 外部世界与单目深度完成统一旋转/平移/尺度变换。
- [ ] target/render 的地标、视差、遮挡和运镜方向逐帧对齐。
- [ ] 不只检查首帧；不只检查矩阵形状。
- [ ] 没有用 `pcd.ply` 代替正式 renderer 做精确几何验收。

### D. 主体中心样本

- [ ] `subject_pixel` 基于 EXIF 校正后的交付原图。
- [ ] 点击在主体内部，深度邻域没有跨到背景。
- [ ] source/render depth、camera/world point 和初始 c2w 合理。
- [ ] 逐帧 K 与实际 render 相同，变焦没有只保存单一 K。

### E. 上模型前

- [ ] 用同尺寸、同帧数设置生成一个全新的缓存目录。
- [ ] `prepare_static_cache.py` 成功生成一个样本且缓存校验通过。
- [ ] `train_static_lora.py --validate_only` 通过；注意它不执行前反向。
- [ ] 在目标设备完成一个训练 step、adapter 保存和严格重新加载。
- [ ] 固定参考图、轨迹、prompt、seed，对比无 LoRA 与有 LoRA 推理。
- [ ] 分别评价相机精度、主体自主运动、背景稳定和遮挡补全，不把所有像素变化都视为错误。

---

## 11. 910B2 和训练边界的补充提醒

- 当前静态微调入口使用 Wan2.1 I2V 14B 和 Uni3C 控制器，不是 Wan2.2 5B 的通用训练脚本。
- `cam_render.py --device npu` 在选定 NPU 上执行 MoGe-3 dense 深度/内参、CarveKit 分割、点云反投影、主体 lifting、相机轨迹、纯 PyTorch splatting 和 ray embedding；不要求 PyTorch3D，也不会静默回退到 Depth-Pro 或其他模型。`--renderer_backend auto` 在 CUDA 上仍使用原 PyTorch3D。纯 PyTorch 回退的边缘合成和 mask 不保证与 CUDA 逐像素一致，批量生成前应做 A/B。
- MoGe-3 NPU 路径使用 `src/moge_npu.py` 隔离未使用的稀疏 refiner import，严格加载 dense 权重，使用 FP32 且 `refine_steps=0`。安装固定源码包时必须执行 `pip install --no-deps -r requirements-moge-npu.txt`，不要替换 Ascend 的 torch/torchvision/torch_npu 或 NumPy。PIL、JSON、PLY、视频解码与 FFmpeg 编码是主机文件 I/O；模型 CPU offload 是推理显存选项，二者都不等于把深度、分割或几何计算回退到 CPU。
- NPU 不支持 `depth-pro`/`moge2-vitl`，这些请求会明确报错；不会因为缺少 Triton、FlexGEMM、xFormers 或 PyTorch3D 而假装完成 NPU 预处理。完整安装、真实权重 smoke test、5/17 帧预处理与训练验证命令见 [910B2 指南](ascend_static_scene.md)。人体动作驱动 `alignment.py`/SMPL mesh renderer 不在本次静态 NPU 适配范围内。
- 17 帧、小 `max_area`、batch size 1 是首次 smoke test 建议，不是显存或质量达标保证。
- `--resume_lora` 只恢复 adapter 权重，不恢复 optimizer、scheduler 和精确 step。
- 本仓库的静态训练/昇腾扩展在当前开发环境中未完成真实 910B2 数值和端到端验证。文档中的坐标约定来自代码核对，不代表训练收敛、HBM 足够或最终静态质量已经达标。

完整环境命令、缓存恢复和 LoRA 推理见 [ascend_static_scene.md](ascend_static_scene.md)。
