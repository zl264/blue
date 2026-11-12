#!/usr/bin/env python3
"""使用 RIFE 4.22.lite 在昇腾 NPU 上将视频插帧为 2 倍帧率。

模型结构适配自 Practical-RIFE / ECCV2022-RIFE（MIT License）。
仅使用 PyTorch、torch_npu、NumPy 和 FFmpeg，不依赖其他推理框架。
Copyright (c) 2021 Zhewei Huang

仅下载模型（不需要安装 PyTorch）：
    python rife_npu_2x.py --download-only

正式处理：
    python rife_npu_2x.py input.mp4 -o output.mp4
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


MODEL_SIZE = 21_429_303
MODEL_SHA256 = "8f6fb9105ba9e946762ee7190acbca3ca1cf14193eb81ca0955d492fb8558692"
MODEL_REPO_PATH = "elfgum/RIFE-4.22.lite/resolve/main/flownet.pkl"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "rife-4.22-lite"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_model_file(path):
    return (
        path.is_file()
        and path.stat().st_size == MODEL_SIZE
        and file_sha256(path) == MODEL_SHA256
    )


def model_urls():
    custom_url = os.environ.get("RIFE_MODEL_URL")
    hf_endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    urls = []
    if custom_url:
        urls.append(custom_url)
    if hf_endpoint:
        urls.append(f"{hf_endpoint}/{MODEL_REPO_PATH}")
    urls.extend(
        [
            f"https://huggingface.co/{MODEL_REPO_PATH}",
            f"https://hf-mirror.com/{MODEL_REPO_PATH}",
        ]
    )
    return list(dict.fromkeys(urls))


def download_once(url, part_path):
    offset = part_path.stat().st_size if part_path.exists() else 0
    if offset >= MODEL_SIZE:
        part_path.unlink()
        offset = 0

    headers = {"User-Agent": "rife-npu-2x/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        append = bool(offset and status == 206)
        if not append:
            offset = 0
        mode = "ab" if append else "wb"
        next_report = offset + 4 * 1024 * 1024
        with part_path.open(mode) as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                offset += len(chunk)
                if offset >= next_report:
                    print(
                        f"\r已下载 {offset / 1048576:.1f}/{MODEL_SIZE / 1048576:.1f} MiB",
                        end="",
                        flush=True,
                    )
                    next_report = offset + 4 * 1024 * 1024
    print(f"\r已下载 {offset / 1048576:.1f}/{MODEL_SIZE / 1048576:.1f} MiB")


def prepare_model(model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "flownet.pkl"
    if model_path.exists():
        if valid_model_file(model_path):
            print(f"模型已就绪: {model_path}")
            return model_path
        raise RuntimeError(f"已有模型校验失败，请删除后重试: {model_path}")

    part_path = model_dir / "flownet.pkl.part"
    if part_path.exists() and valid_model_file(part_path):
        os.replace(part_path, model_path)
        print(f"模型已就绪: {model_path}")
        return model_path

    errors = []
    for url in model_urls():
        for attempt in range(2):
            try:
                print(f"下载 RIFE 4.22.lite: {url}")
                download_once(url, part_path)
                if part_path.stat().st_size != MODEL_SIZE:
                    raise RuntimeError(
                        f"文件大小错误: {part_path.stat().st_size}，应为 {MODEL_SIZE}"
                    )
                if file_sha256(part_path) != MODEL_SHA256:
                    raise RuntimeError("模型 SHA256 校验失败")
                os.replace(part_path, model_path)
                print(f"模型准备完成: {model_path}")
                return model_path
            except Exception as error:
                errors.append(f"{url}（第 {attempt + 1} 次）: {error}")
                print(f"下载失败: {error}", file=sys.stderr)

    details = "\n".join(errors)
    raise RuntimeError(
        "模型下载失败，已保留 .part 文件供下次断点续传。\n"
        "也可设置 RIFE_MODEL_URL 为可用的 flownet.pkl 直链。\n"
        f"{details}"
    )


def parse_fps(value):
    from fractions import Fraction

    if not value or value == "0/0":
        return None
    try:
        fps = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if fps > 0 else None


def probe_video(ffprobe, input_path):
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffprobe 读取输入失败:\n{result.stderr.strip()}")

    import json

    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError("输入文件中没有视频流")
    stream = streams[0]
    fps = parse_fps(stream.get("avg_frame_rate")) or parse_fps(
        stream.get("r_frame_rate")
    )
    if fps is None:
        raise RuntimeError("无法确定输入帧率")

    nb_frames = stream.get("nb_frames", "")
    total_frames = int(nb_frames) if nb_frames and nb_frames.isdigit() else None
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "total_frames": total_frames,
    }


def run_npu(args, model_path, info, output_path, ffmpeg):
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError(
            "缺少 PyTorch 或 NumPy，请先安装与 CANN 匹配的 PyTorch、torch_npu 和 numpy"
        ) from error

    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "未安装 torch_npu。请使用昇腾官方与当前 CANN/PyTorch 匹配的版本，"
            "不要用普通 CUDA 版 PyTorch 替代"
        ) from error

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu 已导入，但 NPU 不可用，请检查驱动、CANN 和环境变量")

    import torch.nn as nn
    import torch.nn.functional as F

    grid_cache = {}

    def warp(ten_input, ten_flow):
        key = (
            str(ten_flow.device),
            ten_flow.dtype,
            ten_flow.shape[0],
            ten_flow.shape[2],
            ten_flow.shape[3],
        )
        if key not in grid_cache:
            horizontal = torch.linspace(
                -1.0,
                1.0,
                ten_flow.shape[3],
                device=ten_flow.device,
                dtype=ten_flow.dtype,
            ).view(1, 1, 1, ten_flow.shape[3])
            vertical = torch.linspace(
                -1.0,
                1.0,
                ten_flow.shape[2],
                device=ten_flow.device,
                dtype=ten_flow.dtype,
            ).view(1, 1, ten_flow.shape[2], 1)
            horizontal = horizontal.expand(
                ten_flow.shape[0], -1, ten_flow.shape[2], -1
            )
            vertical = vertical.expand(
                ten_flow.shape[0], -1, -1, ten_flow.shape[3]
            )
            grid_cache[key] = torch.cat([horizontal, vertical], dim=1)

        flow = torch.cat(
            [
                ten_flow[:, 0:1] / ((ten_input.shape[3] - 1.0) / 2.0),
                ten_flow[:, 1:2] / ((ten_input.shape[2] - 1.0) / 2.0),
            ],
            dim=1,
        )
        grid = (grid_cache[key] + flow).permute(0, 2, 3, 1)
        return F.grid_sample(
            ten_input,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
        )

    class ResConv(nn.Module):
        def __init__(self, channels, dilation=1):
            super().__init__()
            self.conv = nn.Conv2d(
                channels,
                channels,
                3,
                1,
                dilation,
                dilation=dilation,
                groups=1,
            )
            self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
            self.relu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            return self.relu(self.conv(x) * self.beta + x)

    class IFBlock(nn.Module):
        def __init__(self, in_planes, channels):
            super().__init__()
            self.conv0 = nn.Sequential(
                conv(in_planes, channels // 2, 3, 2, 1),
                conv(channels // 2, channels, 3, 2, 1),
            )
            self.convblock = nn.Sequential(*[ResConv(channels) for _ in range(8)])
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(channels, 4 * 13, 4, 2, 1),
                nn.PixelShuffle(2),
            )

        def forward(self, x, flow=None, scale=1.0):
            x = F.interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            )
            if flow is not None:
                flow = (
                    F.interpolate(
                        flow,
                        scale_factor=1.0 / scale,
                        mode="bilinear",
                        align_corners=False,
                    )
                    / scale
                )
                x = torch.cat((x, flow), dim=1)
            feat = self.convblock(self.conv0(x))
            tmp = F.interpolate(
                self.lastconv(feat),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            )
            return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
            self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
            self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
            self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
            self.relu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            x = self.relu(self.cnn0(x))
            x = self.relu(self.cnn1(x))
            x = self.relu(self.cnn2(x))
            return self.cnn3(x)

    class IFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block0 = IFBlock(7 + 8, 192)
            self.block1 = IFBlock(8 + 4 + 8 + 8, 128)
            self.block2 = IFBlock(8 + 4 + 8 + 8, 64)
            self.block3 = IFBlock(8 + 4 + 8 + 8, 32)
            self.encode = Head()

        def forward(self, x, timestep=0.5, scale_list=None):
            scale_list = scale_list or [8, 4, 2, 1]
            img0, img1 = x[:, :3], x[:, 3:]
            if not torch.is_tensor(timestep):
                timestep = (x[:, :1].clone() * 0 + 1) * timestep
            else:
                timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

            f0 = self.encode(img0)
            f1 = self.encode(img1)
            flow = None
            mask = None
            feat = None
            warped_img0 = img0
            warped_img1 = img1
            blocks = [self.block0, self.block1, self.block2, self.block3]

            for index, block in enumerate(blocks):
                if flow is None:
                    flow, mask, feat = block(
                        torch.cat((img0, img1, f0, f1, timestep), dim=1),
                        None,
                        scale_list[index],
                    )
                else:
                    wf0 = warp(f0, flow[:, :2])
                    wf1 = warp(f1, flow[:, 2:4])
                    flow_delta, mask, feat = block(
                        torch.cat(
                            (
                                warped_img0,
                                warped_img1,
                                wf0,
                                wf1,
                                timestep,
                                mask,
                                feat,
                            ),
                            dim=1,
                        ),
                        flow,
                        scale_list[index],
                    )
                    flow = flow + flow_delta
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])

            mask = torch.sigmoid(mask)
            return warped_img0 * mask + warped_img1 * (1 - mask)

    class RIFE(nn.Module):
        def __init__(self):
            super().__init__()
            self.flownet = IFNet()

        def forward(self, img0, img1, scale):
            height, width = img0.shape[2:]
            pad_unit = max(32, int(32 / scale))
            padded_height = ((height - 1) // pad_unit + 1) * pad_unit
            padded_width = ((width - 1) // pad_unit + 1) * pad_unit
            padding = (0, padded_width - width, 0, padded_height - height)
            imgs = torch.cat((F.pad(img0, padding), F.pad(img1, padding)), dim=1)
            result = self.flownet(
                imgs,
                timestep=0.5,
                scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale],
            )
            return result[:, :, :height, :width]

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    model = RIFE()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError("模型文件格式错误：顶层不是 state_dict")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    expected_keys = model.flownet.state_dict().keys()
    inference_state = {key: value for key, value in state.items() if key in expected_keys}
    model.flownet.load_state_dict(inference_state, strict=True)
    ignored = len(state) - len(inference_state)
    if ignored:
        print(f"已忽略 {ignored} 个仅训练使用的教师网络参数")
    model = model.to(device=device, dtype=dtype).eval()

    width, height = info["width"], info["height"]
    frame_bytes = width * height * 3
    output_fps = info["fps"] * 2
    input_path = str(args.input.resolve())

    decoder_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-noautorotate",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    encoder_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if args.overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-framerate",
        str(output_fps),
        "-i",
        "pipe:0",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map_metadata",
        "1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    def read_exact(pipe, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = pipe.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def to_npu(raw):
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=device, dtype=dtype).div_(255.0)

    def to_bytes(tensor):
        array = (
            tensor.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .cpu()
            .contiguous()
            .numpy()
        )
        return array.tobytes()

    decoder = None
    encoder = None
    frames_in = 0
    frames_out = 0
    started = time.perf_counter()
    print(
        f"NPU: {device} | dtype: {args.dtype} | "
        f"{width}x{height} | {info['fps']} -> {output_fps} fps"
    )

    try:
        encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE)
        decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE)
        previous_raw = read_exact(decoder.stdout, frame_bytes)
        if not previous_raw:
            raise RuntimeError("没有从输入视频解码到任何帧")
        if len(previous_raw) != frame_bytes:
            raise RuntimeError("首帧解码数据不完整")

        encoder.stdin.write(previous_raw)
        frames_in = 1
        frames_out = 1
        previous = to_npu(previous_raw)

        with torch.inference_mode():
            while True:
                current_raw = read_exact(decoder.stdout, frame_bytes)
                if not current_raw:
                    break
                if len(current_raw) != frame_bytes:
                    raise RuntimeError("解码到不完整的视频帧")

                if current_raw == previous_raw:
                    encoder.stdin.write(previous_raw)
                else:
                    current = to_npu(current_raw)
                    middle = model(previous, current, args.scale)
                    encoder.stdin.write(to_bytes(middle))
                    previous = current
                encoder.stdin.write(current_raw)
                frames_in += 1
                frames_out += 2
                previous_raw = current_raw

                if frames_in % 10 == 0:
                    total = info["total_frames"] or "?"
                    print(
                        f"\r已处理 {frames_in}/{total} 帧，输出 {frames_out} 帧",
                        end="",
                        flush=True,
                    )

        encoder.stdin.close()
        decoder.stdout.close()
        encoder_code = encoder.wait()
        decoder_code = decoder.wait()
        if decoder_code:
            raise RuntimeError(f"FFmpeg 解码失败，退出码 {decoder_code}")
        if encoder_code:
            raise RuntimeError(
                f"FFmpeg 编码失败，退出码 {encoder_code}；"
                "请检查 libx264 是否可用，以及原音频编码是否能放入输出容器"
            )
    except Exception:
        for process in (decoder, encoder):
            if process is not None and process.poll() is None:
                process.kill()
        for process in (decoder, encoder):
            if process is not None:
                process.wait()
        output_path.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    expected_frames = 2 * frames_in - 1
    if frames_out != expected_frames:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"输出帧数异常: {frames_out}，预期 {expected_frames}")
    print(
        f"\n完成: {frames_in} -> {frames_out} 帧，耗时 {elapsed:.2f} 秒\n"
        f"输出: {output_path}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="RIFE 4.22.lite + 昇腾 NPU 视频二倍插帧（保留原音频）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python rife_npu_2x.py --download-only\n"
            "  python rife_npu_2x.py input.mp4\n"
            "  python rife_npu_2x.py input.mp4 -o result.mp4 --dtype fp16\n"
            "  python rife_npu_2x.py input.mp4 --scale 0.5  # 降低显存占用"
        ),
    )
    parser.add_argument("input", nargs="?", type=Path, help="输入视频")
    parser.add_argument("-o", "--output", type=Path, help="输出视频，默认 <输入名>_2x.mp4")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="模型缓存目录")
    parser.add_argument("--download-only", action="store_true", help="只下载并校验模型")
    parser.add_argument("--device", type=int, default=0, help="NPU 编号，默认 0")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16", help="推理精度")
    parser.add_argument(
        "--scale",
        type=float,
        choices=(1.0, 0.5, 0.25),
        default=1.0,
        help="RIFE 推理尺度；显存不足时可用 0.5，质量会下降",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.download_only:
        prepare_model(args.model_dir)
        return
    if args.input is None:
        parser.error("请输入视频路径，或使用 --download-only")
    if not args.input.is_file():
        parser.error(f"输入文件不存在: {args.input}")

    output_path = args.output or args.input.with_name(f"{args.input.stem}_2x.mp4")
    if args.input.resolve() == output_path.resolve():
        parser.error("输出文件不能覆盖输入文件")
    if output_path.exists() and not args.overwrite:
        parser.error(f"输出已存在: {output_path}；如需覆盖请添加 --overwrite")
    if not output_path.parent.is_dir():
        parser.error(f"输出目录不存在: {output_path.parent}")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("未找到 ffmpeg/ffprobe，请安装 FFmpeg 并加入 PATH")

    info = probe_video(ffprobe, args.input)
    if info["width"] % 2 or info["height"] % 2:
        raise RuntimeError("为兼容 yuv420p/H.264，输入宽度和高度必须为偶数")

    model_path = prepare_model(args.model_dir)
    run_npu(args, model_path, info, output_path, ffmpeg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户取消", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"\n错误: {error}", file=sys.stderr)
        sys.exit(1)
