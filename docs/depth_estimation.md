# Uni3C 可商用深度模型调研与 MoGe-3 接入

调研日期：**2026-11-16**。许可证结论基于官方仓库和 Hugging Face 模型卡，不构成法律意见；商业发布前仍应固定代码/权重版本、保留许可证，并检查所有间接依赖和输入数据许可。

## 1. 结论

**效果优先：推荐 MoGe-3 ViT-G；综合效果、显存与部署成本：默认采用 MoGe-3 ViT-L。**

原因不是“模型越新越好”，而是它同时满足：

1. 官方代码和 `moge-3-vitl/vitg` 权重模型卡均标注 **MIT**，允许商业使用；
2. 输出与 Uni3C 直接相关的米制相机 z 深度、OpenCV 点图、归一化针孔 K 和有效 mask；
3. 官方九项零样本均值显示全局/局部几何优于 Depth-Pro、MoGe-2、DA3 和 UniDepthV2；
4. 可设置已知 FOV，且保持输入/输出图像网格对齐。

当前 `cam_render.py` 和人体/世界点云对齐入口 `alignment.py` 均默认 `--depth_model moge3-vitl`；需要追求官方表格中更强全局结果时切换 `moge3-vitg`。Depth-Pro 保留为 `--depth_model depth-pro`。两条入口共用 `src/depth_estimation.py`，避免统一人体/相机控制仍静默使用旧后端。

**这不是已验证的最终画质结论。** 官方深度指标不等于 Uni3C 运镜视频质量；必须在同一批参考图、轨迹、prompt 和 seed 下做端到端 A/B。

## 2. 候选与商业许可

| 模型 | 商业使用判断 | 与 Uni3C 的适配 | 结论 |
|---|---|---|---|
| **MoGe-3 ViT-L / ViT-G** | 代码及权重模型卡 MIT | 原生 metric z、OpenCV 点图、K、mask | **首选** |
| **MoGe-2 ViT-L** | 代码及权重模型卡 MIT | 同类输出，无 MoGe-3 稀疏细化 | 依赖/性能备选 |
| **DA3METRIC-LARGE** | 代码及权重 Apache-2.0 | 单目 metric 系列，但官方公式需要 focal：`focal * output / 300` | 轻依赖备选，需另配 K |
| Metric3Dv2 | 官方代码 BSD-2-Clause | metric depth，通常需要相机规范化与焦距输入 | 成熟但较旧 |
| Depth-Pro | Apple 自定义许可，需按原文审查 | 官方 Uni3C 已适配，输出 metric z 和 focal | 保留回退基线 |
| UniDepthV2 | **CC BY-NC 4.0** | 直接 metric z/K，指标优秀 | **排除商业用途** |
| DA3 Giant/Large/Nested | **CC BY-NC 4.0** | 功能强，Nested 直接米制输出 | **不能因“DA3 Apache”而误用这些权重** |
| DA3 Base/Small | Apache-2.0 | 非专用 metric 型号 | 不是效果优先选择 |

注意：DA3 仓库根 LICENSE 是 Apache-2.0，但官方 Model Zoo 给不同权重标注了不同许可。**代码许可不能替代权重许可。**

## 3. 官方效果证据

MoGe-3 官方项目页给出的九项零样本均值如下。数值越低/越高以箭头为准；“Metric Depth Rel”是米制尺度结果，不是 Uni3C 视频指标。

| 方法 | 局部深度 Rel ↓ | 局部点 Rel ↓ | 全局相对深度 Rel ↓ | 全局相对点 Rel ↓ | Metric Depth Rel ↓ | 边界 F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Depth-Pro | 4.15 | 3.01 | 8.10 | 11.1 | 29.0 | 16.3 |
| Depth Anything 3 | 10.5 | 7.39 | 7.89 | 10.6 | 17.4 | 6.61 |
| UniDepthV2（非商用） | 5.60 | 4.60 | 6.98 | 10.2 | 23.7 | 14.5 |
| MoGe-2 | 3.81 | 3.19 | 5.98 | 8.73 | 15.6 | 15.6 |
| **MoGe-3 ViT-L** | 3.76 | **2.79** | 5.52 | 7.91 | **15.0** | 16.0 |
| **MoGe-3 ViT-G** | **3.66** | 2.80 | **4.80** | **7.43** | 15.8 | **17.3** |

解读：

- **ViT-G** 的全局、局部细节和边界整体最好，但 checkpoint 约 **5.00 GB**（1.25B 参数），不适合作为默认低成本后端。
- **ViT-L** 的米制深度均值和局部点误差更好，且模型更小，是合理默认。
- MoGe-3 并非每项都绝对最优：ViT-L 的边界 F1 略低于 Depth-Pro/MoGe-2，ViT-G 的米制 Rel 略高于 ViT-L。
- 这些是 MoGe 官方结果，不是独立复现，也没有证明 Uni3C 最终视频一定更好。

官方称三步稀疏细化额外约 121 ms，但这依赖硬件与设置，本机未复现。MoGe 的输入/输出 H,W 一致，但内部仍以 token 分辨率解码并重采样到输入尺寸，**不能宣称原生逐像素高分辨率输出**。

## 4. 已实现的后端接口

```text
resized RGB [3,H,W], [0,1]
  -> src/depth_estimation.py
  -> positive finite camera-z [1,1,H,W]
  -> pixel K [3,3]（允许 fx != fy）
  -> Uni3C 原点云、主体反投影和轨迹
```

新增文件 `src/depth_estimation.py`：

- 懒加载 MoGe，Depth-Pro 不被强制依赖 MoGe 环境；
- 使用 `moge.model.v3.MoGeModel.from_pretrained(...)`；
- 输出采用 `prediction["depth"]`，不是点图的欧氏长度；
- 将归一化 K 的第 0/1 行分别乘 W/H；
- `apply_mask=False` 防止无效值进入后处理，随后用保守远值替换非有限/非正深度；
- 将无效像素数量写入 `cam_info.json`；
- 不把填充区域宣称为真实测量深度。

`cam_render.py` 和 `alignment.py` 共用的新参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--depth_model` | `moge3-vitl` | `moge3-vitl/moge3-vitg/moge2-vitl/depth-pro` |
| `--depth_checkpoint` | 官方 HF 仓库 | 可选本地路径或模型仓库 |
| `--depth_refine_steps` | `3` | MoGe-3 稀疏细化次数，`0` 可关闭 |
| `--depth_fp16` | 关闭 | 混合精度；默认 FP32 优先几何稳定性 |

深度推理后释放估计器，再加载前景分割/GeoCalib 模型，避免深度模型与这些预处理模型长期同时驻留 GPU。主体中心轨迹仍使用同一深度和 K，接口不变。

`alignment.py` 的环境点云和人体关键点也直接使用该后端返回的相机 z 与完整 K；GeoCalib 保留官方标量焦距先验近似，但点云和 PyTorch3D 渲染不再用标量焦距覆盖 MoGe 的 `fx/fy`。深度填充像素数同样写入其 `cam_info.json`。

## 5. 安装与依赖冲突（重要）

官方 MoGe-3 安装：

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

**不要在未验证的 Uni3C 生产环境中直接执行。** 截至调研时：

- MoGe-3 的 `pyproject.toml` 要求 `numpy>=2`；
- Uni3C `requirements.txt` 固定 `numpy==1.26.4`；
- MoGe-3 依赖 FlexGEMM/Triton、`utils3d_moge` 等；
- Uni3C 使用 PyTorch 2.6/CUDA 12.4，而 MoGe 当前 uv 默认 CUDA 13.0。

推荐在 GPU 机器上为 MoGe 建独立环境，先导出与 RGB 对齐的 depth/K，再由 Uni3C 环境消费；或者建立一个新的联合环境，用 `pip check` 和实际导入验证 NumPy 2 对 Uni3C、PyTorch3D、CarveKit 等的影响。当前代码提供统一后端，**没有伪造已解决的依赖兼容**。

FlexGEMM 等间接依赖的许可证也应在商业发布时随 SBOM 一起审核；本轮 GitHub API 达到限流，未能完整核验每个间接依赖许可证，不能声称整个依赖树均已完成法律审查。

## 6. 使用

```bash
# 默认 MoGe-3 ViT-L
python cam_render.py \
  --reference_image data/demo/flux-0.png \
  --output_path outputs/moge3l \
  --subject_pixel 520 360 \
  --d_phi -60 --d_r 0.8

# 效果优先 ViT-G
python cam_render.py \
  --reference_image data/demo/flux-0.png \
  --output_path outputs/moge3g \
  --depth_model moge3-vitg \
  --subject_pixel 520 360 --d_phi -60

# 原始 Depth-Pro 回退
python cam_render.py \
  --reference_image data/demo/flux-0.png \
  --output_path outputs/depthpro \
  --depth_model depth-pro \
  --subject_pixel 520 360 --d_phi -60
```

`cam_control.py`、`uni3c_inference.py` 和 PCDController 命令不变。人体/世界点云对齐同样可以选择后端，例如：

```bash
python alignment.py \
  --input_image data/demo_uni3c/reference.png \
  --input_path outputs/demo_uni3c \
  --output_path outputs/demo_uni3c_aligned \
  --depth_model moge3-vitl \
  --traj_type free1
```

## 7. 必须做的 GPU A/B 验收

1. 固定至少 20 张室内/室外、人物/物体、近景/远景、天空/反光/细边界图片。
2. 分别用 Depth-Pro、MoGe-3-L、MoGe-3-G 渲染相同轨迹；检查深度边缘、主体三维位置、空洞比例及 `filled_pixels`。
3. 用同一 prompt、seed 和 PCDController 权重生成，记录运镜精度、主体身份保持、背景稳定性、空洞伪影和人工偏好。
4. 记录深度模型显存、加载时间、推理时间、细化次数与 FP32/FP16 差异。
5. 如果 MoGe-3 深度边缘更好但最终视频没有稳定改善，不应仅凭 benchmark 更换默认模型。

## 8. 当前验证边界

本轮完成官方文档/许可证/接口调研和静态代码接入；**未安装 MoGe、未下载约数 GB 权重、未运行深度推理或 Uni3C GPU 渲染**。本机没有可用 PyTorch 环境，且外部测试执行受权限限制。因此需要在目标 GPU 机器完成上述验收后再认定替换成功。

## 官方来源

- MoGe 代码与安装：https://github.com/microsoft/MoGe
- MoGe-3 项目与指标：https://qft-333.github.io/moge3page/
- MoGe-3 ViT-L 权重：https://huggingface.co/Ruicheng/moge-3-vitl
- MoGe-3 ViT-G 权重：https://huggingface.co/Ruicheng/moge-3-vitg
- MoGe-2 论文：https://arxiv.org/abs/2507.02546
- DA3 模型许可与 focal 公式：https://github.com/ByteDance-Seed/Depth-Anything-3
- DA3METRIC-LARGE：https://huggingface.co/depth-anything/DA3METRIC-LARGE
- UniDepthV2：https://github.com/lpiccinelli-eth/UniDepth
- Metric3D：https://github.com/YvanYin/Metric3D
- Depth-Pro：https://github.com/apple/ml-depth-pro
