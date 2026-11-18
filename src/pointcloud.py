import contextlib
import os
import sys

import einops
import torch

from src.utils import points_padding


def __getattr__(name):
    """Preserve the legacy alignment import without eager PyTorch3D imports."""
    if name == "PointsZbufRenderer":
        from pytorch3d.renderer import PointsRenderer

        class PointsZbufRenderer(PointsRenderer):
            def forward(self, point_clouds, **kwargs):
                fragments = self.rasterizer(point_clouds, **kwargs)
                radius = self.rasterizer.raster_settings.radius
                dists2 = fragments.dists.permute(0, 3, 1, 2)
                weights = 1 - dists2 / (radius * radius)
                images = self.compositor(
                    fragments.idx.long().permute(0, 3, 1, 2), weights,
                    point_clouds.features_packed().permute(1, 0), **kwargs,
                )
                return images.permute(0, 2, 3, 1), fragments.zbuf

        return PointsZbufRenderer
    raise AttributeError(name)


def get_boundaries_mask(disparity, sobel_threshold=0.3):
    def sobel_filter(disp, mode="sobel", beta=10.0):
        # Kornia's spatial gradient is equivalent to padding + a fixed depthwise
        # convolution. Keeping the small formula here avoids importing Kornia or
        # a renderer backend during NPU preprocessing.
        import torch.nn.functional as F

        if mode != "sobel":
            raise ValueError("Only Sobel gradients are supported")
        kernels = torch.tensor(
            [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
             [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]],
            dtype=disp.dtype, device=disp.device,
        ).unsqueeze(1)
        gradient = F.conv2d(F.pad(disp, (1, 1, 1, 1), mode="reflect"), kernels)
        sobel_mag = torch.sqrt(gradient[:, 0:1] ** 2 + gradient[:, 1:2] ** 2)
        alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

        return alpha

    sobel_beta = 10.0
    normalized_disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    return sobel_filter(normalized_disparity, "sobel", beta=sobel_beta) < sobel_threshold


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def contract_depth(depth, contract=8.0):
    """Return the finite far-depth transform used by the point renderer.

    This is shared with subject lifting so orbit radii and actual rendered
    geometry use exactly the same depth scale. Source camera-z remains the input.
    """
    mid_depth = torch.median(depth.reshape(-1), dim=0)[0] * contract
    return torch.where(
        depth > mid_depth,
        2 * mid_depth - mid_depth ** 2 / (depth + 1e-6),
        depth,
    )


def point_rendering(K, w2cs, depth, image, raster_settings, device,
                    background_color=[0, 0, 0], sobel_threshold=0.35, contract=8.0,
                    sam_mask=None, renderer_backend="auto", torch_splat_radius=1,
                    torch_chunk_size=32768):
    """
    only support batchsize=1
    :param K: [F,3,3]
    :param w2cs: [F,4,4] opencv
    :param depth: [1,1,H,W]
    :param images: [1,3,H,W]
    :param background_color: [-1,-1,-1]~[1,1,1]
    :param raster_settings:
    :param sam_mask: [1,1,H,W] 0 or 1
    :return: render_rgbs, render_masks
    """
    nframe = w2cs.shape[0]
    _, _, h, w = image.shape

    # depth contract
    depth = depth.to(device).clone()
    K = K.to(device)
    w2cs = w2cs.to(device)
    image = image.to(device)
    c2ws = w2cs.inverse()

    if depth.max() == 0:
        render_rgbs = torch.zeros((nframe, 3, h, w), device=device, dtype=torch.float32)
        render_masks = torch.ones((nframe, 1, h, w), device=device, dtype=torch.float32)
    else:
        depth = contract_depth(depth, contract=contract)

        point_depth = einops.rearrange(depth[0], "c h w -> (h w) c")
        disp = 1 / (depth + 1e-7)
        boundary_mask = get_boundaries_mask(disp, sobel_threshold=sobel_threshold)

        x = torch.arange(w, device=device).float() + 0.5
        y = torch.arange(h, device=device).float() + 0.5
        points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        points = einops.rearrange(points, "w h c -> (h w) c")
        points_3d = (c2ws[0] @ points_padding((K[0].inverse() @ points_padding(points).T).T * point_depth).T).T[:, :3]

        colors = einops.rearrange(image[0], "c h w -> (h w) c")

        # boundary mask: [1,1,H,W] -> [P]
        boundary_mask = boundary_mask.reshape(-1)
        if sam_mask is not None:
            sam_mask = sam_mask.reshape(-1)
            boundary_mask[sam_mask == True] = True

        points_3d = points_3d[boundary_mask == False]

        if points_3d.shape[0] <= 8:
            render_rgbs = torch.zeros((nframe, 3, h, w), device=device, dtype=torch.float32)
            render_masks = torch.ones((nframe, 1, h, w), device=device, dtype=torch.float32)
            render_rgbs[0:1] = image
            render_masks[0:1] = 0
            return render_rgbs, render_masks

        colors = colors[boundary_mask == False]

        if renderer_backend == "auto":
            renderer_backend = "torch" if torch.device(device).type in ("npu", "cpu") else "pytorch3d"
        if renderer_backend == "torch":
            from src.pointcloud_torch import render_points_torch

            print(
                f"Using pure PyTorch point splatting on {device}; "
                f"radius={torch_splat_radius}, chunk={torch_chunk_size}. "
                "This is not bit-identical to PyTorch3D compositing."
            )
            render_rgbs, render_masks = render_points_torch(
                points_3d, colors, K, w2cs, h, w,
                background_color=background_color,
                splat_radius=torch_splat_radius, chunk_size=torch_chunk_size,
                progress=True,
            )
        elif renderer_backend == "pytorch3d":
            # Import CUDA-oriented renderer dependencies only on the legacy path.
            # Importing this module on Ascend must not initialize PyTorch3D.
            from pytorch3d.renderer import (
                PointsRenderer, PointsRasterizer, AlphaCompositor, PerspectiveCameras,
            )
            from pytorch3d.structures import Pointclouds

            class PointsZbufRenderer(PointsRenderer):
                def forward(self, point_clouds, **kwargs):
                    fragments = self.rasterizer(point_clouds, **kwargs)
                    r = self.rasterizer.raster_settings.radius
                    dists2 = fragments.dists.permute(0, 3, 1, 2)
                    weights = 1 - dists2 / (r * r)
                    images = self.compositor(
                        fragments.idx.long().permute(0, 3, 1, 2), weights,
                        point_clouds.features_packed().permute(1, 0), **kwargs,
                    )
                    return images.permute(0, 2, 3, 1), fragments.zbuf

            point_cloud = Pointclouds(points=[points_3d], features=[colors]).extend(nframe)

            # Convert OpenCV camera-local axes to PyTorch3D's screen convention.
            c2ws[:, :, 0] = -c2ws[:, :, 0]
            c2ws[:, :, 1] = -c2ws[:, :, 1]
            w2cs = c2ws.inverse()

            focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
            principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1)
            image_shapes = torch.tensor(
                [[h, w]], dtype=torch.float32, device=device,
            ).repeat(nframe, 1)
            cameras = PerspectiveCameras(
                focal_length=focal_length, principal_point=principal_point,
                R=c2ws[:, :3, :3], T=w2cs[:, :3, -1], in_ndc=False,
                image_size=image_shapes, device=device,
            )
            renderer = PointsZbufRenderer(
                rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
                compositor=AlphaCompositor(background_color=background_color),
            )
            with suppress_stdout_stderr():
                render_rgbs, zbuf = renderer(point_cloud)
            render_masks = (zbuf[..., 0:1] == -1).float()
            render_rgbs = einops.rearrange(render_rgbs, "f h w c -> f c h w")
            render_masks = einops.rearrange(render_masks, "f h w c -> f c h w")
        else:
            raise ValueError("renderer_backend must be auto, pytorch3d or torch")

    # replace the first frame
    render_rgbs[0:1] = image
    render_masks[0:1] = 0

    return render_rgbs, render_masks
