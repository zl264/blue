# 主体中心运镜扩展

本扩展为 Uni3C 增加：**给定参考图上的主体像素坐标，以该主体三维位置为中心进行环绕、俯仰、升降、推近拉远、变焦及组合运镜。**

修改集中在深度反投影、相机轨迹与渲染前处理层，**不修改 PCDController 网络结构或权重格式**。

## 1. 快速使用

```bash
python cam_render.py \
  --reference_image "data/demo/flux-0.png" \
  --output_path "outputs/subject_orbit" \
  --subject_pixel 520 360 \
  --traj_type custom \
  --d_phi -90 \
  --d_theta -10 \
  --d_r 0.8 \
  --focal_length 1.1
```

`520 360` 仅为示例，需替换为实际主体中心坐标。生成阶段不变：

```bash
CUDA_VISIBLE_DEVICES=0 python cam_control.py \
  --reference_image "data/demo/flux-0.png" \
  --render_path "outputs/subject_orbit" \
  --output_path "outputs/subject_orbit/result.mp4" \
  --prompt "your prompt"
```

不传 `--subject_pixel` 时保持官方原始逻辑。

## 2. 新增参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--subject_pixel U V` | 无 | EXIF 校正后的**原参考图**像素坐标，启用主体中心模式 |
| `--subject_depth Z` | 自动估计 | 可选正数深度覆盖；是渲染深度单位下的相机 z 深度，不是 disparity |
| `--subject_depth_radius` | `2` | 内部渲染分辨率下的深度采样半径，取有效深度中位数；`0` 为单像素 |
| `--subject_look_at_start` | `0.1` | 前 10% 帧从原始朝向平滑转向主体；`0` 表示第 0 帧后立即对准 |
| `--subject_keep_upright` | 关闭 | 使用 `--subject_world_up` 指定的世界上方向 |
| `--subject_world_up X Y Z` | `0 -1 0` | 世界上方向；默认是世界 `-y`（图像向上） |

不开启 `--subject_keep_upright` 时，沿用首相机的上方向，减少初始相机仰角带来的额外滚转。

以下官方参数继续有效：

- `--d_phi`：水平环绕；
- `--d_theta`：垂直环绕；
- `--d_r`：相对首帧主体距离的缩放，小于 `1` 推近，大于 `1` 拉远；
- `--x_offset/y_offset/z_offset`：平移、升降及前后移动，沿用官方相对前景深度的尺度；
- `--focal_length`：变焦；
- `--traj_type`：`custom/free1/.../free5/swing1/swing2/orbit`。

注意：`d_r=1` 保持首帧相机到主体的三维欧氏距离。`d_r` 不是以米为单位的绝对距离。

## 3. 几何与坐标约定

统一采用 OpenCV 相机坐标：x 右、y 下、z 前；位姿为 `c2w`，深度为正相机 z 深度。

### 3.1 点击坐标缩放

原图尺寸为 `(W0,H0)`，Uni3C 内部渲染尺寸为 `(W,H)`。图像只做 resize，不做 crop：

```text
u_render = (u + 0.5) * W / W0 - 0.5
v_render = (v + 0.5) * H / H0 - 0.5
```

渲染器采用像素中心 `(u+0.5,v+0.5)` 反投影。图像 resize 只改变像素和内参，**不乘深度值**。

### 3.2 主体三维提升

```text
P_camera = Z_subject * K^-1 * [u_render+0.5, v_render+0.5, 1]
P_subject = R_initial * P_camera + t_initial
```

`Z_subject` 默认来自与 RGB 对齐的 Depth-Pro 深度图。为与实际点云渲染一致，对超过渲染阈值的远深度，先应用与 `point_rendering` 相同的 depth contraction，再计算主体世界坐标。

如果指定 `--subject_depth Z`，`Z` 必须是应用该渲染变换后的深度单位；图像缩放不会再次缩放它。`subject_info.json` 同时记录源深度和渲染深度，便于核对。

### 3.3 主体中心轨迹

官方轨迹围绕世界原点运动。本扩展先计算：

```text
v_initial = C_initial - P_subject
```

然后对 `v_initial` 应用原有方位角、俯仰角、平移和距离缩放，并加回主体位置。因而首帧相机位置不变，`d_r=1` 时主体半径也不因点击偏心而变化。

朝向采用 look-at：z 轴指向主体，x 轴向右，y 轴向下。第 0 帧完全保留原始位姿；后续帧通过四元数 SLERP 平滑转向主体。世界 up 与视线平行时，使用首相机右轴构造稳定备用方向，避免 NaN。

**平滑转向期间主体逐渐进入画面中心；转向完成后，在无额外平移的组合轨迹中主体投影位于主点。** 首帧不会为了居中而突然旋转。

## 4. 修改文件

| 文件 | 修改内容 |
|---|---|
| `cam_render.py` | 新增 CLI、主体提升、轨迹朝向、元数据输出 |
| `src/subject_camera.py` | 坐标缩放、反投影、look-at、SLERP 和退化处理 |
| `src/utils.py` | `build_cameras(..., orbit_center=...)`，兼容原调用 |
| `src/pointcloud.py` | 提取共享 `contract_depth`，并避免原地修改调用方深度 |
| `src/dataset.py` | 可选读取逐帧 K，匹配变焦渲染 |
| `tests/test_subject_camera.py` | 不依赖模型权重的 NumPy 几何测试 |
| `README.md` | 英文快速说明和示例 |

## 5. 输出与兼容性

仍生成原有：

- `render.mp4`；
- `render_mask.mp4`；
- `pcd.ply`；
- `cam_info.json`。

主体模式额外生成 `subject_info.json`，并在 `cam_info.json` 增加：

- `subject`：原图/渲染坐标、源深度、渲染深度、相机/世界三维点、上方向、初始位姿；
- `per_frame_intrinsic`：每帧真实渲染内参，包含原有 `focal_length` 渐变。

旧字段 `intrinsic` 和 `extrinsic` 保留；旧输出没有 `per_frame_intrinsic` 时仍按官方单 K 逻辑读取。PCDController 的输入接口不变。

## 6. 常见问题与限制

- **主体偏向背景或深度跳变**：检查 `subject_info.json` 的源深度，减小 `--subject_depth_radius`，或点击主体内部。
- **运动尺度异常**：Depth-Pro 深度、三维中心和相机平移共享渲染深度尺度；不要把像素缩放系数乘到深度上。
- **运镜后主体暂时不在中央**：这是默认前 10% 帧平滑转向的结果；可减小 `--subject_look_at_start`，但不建议对偏心主体设为零。
- **大角度出现空洞或背面伪影**：单目首帧点云没有真实背面，仍依赖生成模型补全；本扩展不保证 360° 的真实三维一致性。
- **首帧与原图不一致**：渲染器仍沿用官方首帧替换策略；额外检查深度、分割和参考图预处理是否成功。
- **升降并非固定像素速度**：`y_offset` 沿用官方按前景深度缩放的相对单位，不是图像像素位移。

## 7. 验证状态与后续测试

已完成所有修改 Python 文件的 AST 语法检查与 `git diff --check`。

**尚未完成 NumPy 数值测试、完整 Uni3C 点云渲染和 14B 视频生成。** 当前开发环境的权限策略拒绝执行克隆仓库中的测试脚本，也未安装完整 Uni3C/PyTorch 运行环境。因此不能把上述静态检查视为渲染或生成质量验证。

在完整环境中先运行：

```bash
python -I tests/test_subject_camera.py
```

随后至少完成：

1. 中心与偏心点击各生成一次 `render.mp4/render_mask.mp4`；
2. 检查 `cam_info.json` 中首帧位姿、主体三维点和逐帧 K；
3. 对 `d_phi/d_theta/d_r/y_offset/focal_length` 分别测试，再测试组合轨迹；
4. 用相同 prompt、seed 和输出帧数运行 PCDController，检查主体居中与视频时序稳定性。

本扩展目前未自动提交或推送到远程仓库。
