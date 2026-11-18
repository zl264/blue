"""Device-path and checkpoint tests; requires PyTorch, not MoGe weights."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from src.camera import get_camera_embedding
from src.moge_npu import (
    load_moge3_dense, recover_focal_shift_npu, _normalized_view_plane_uv,
)
from src.subject_camera_torch import aim_cameras_at_subject, lift_subject_to_world


class PortableMoGeTest(unittest.TestCase):
    def test_dense_weights_are_strict_and_sparse_module_is_isolated(self):
        v3 = types.ModuleType("moge.model.v3")
        modules = {
            "moge": types.ModuleType("moge"),
            "moge.model": types.ModuleType("moge.model"),
            "moge.model.v3": v3,
        }
        state = {"model_config": {"refiner": {"channels": 16}}, "model": {}}

        class DenseModel(nn.Module):
            def __init__(self, **config):
                super().__init__()
                if config.get("refiner") is not None:
                    raise RuntimeError("The sparse refiner must not be constructed")
                self.weight = nn.Parameter(torch.zeros(1))
                state["model_config"] = config

            def forward(self, image, **kwargs):
                return {"points": image}

        v3.MoGeModel = DenseModel
        v3.Sparse3DUNet = object
        dense = DenseModel()
        state["model"] = {
            "weight": torch.tensor([2.0]),
            "refiner.unused": torch.tensor([99.0]),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(state, path)
            with mock.patch.dict(sys.modules, modules):
                with mock.patch("importlib.import_module", return_value=v3) as importer:
                    model = load_moge3_dense(path, "cpu")
        self.assertEqual(importer.call_args.args[0], "moge.model.v3")
        self.assertEqual(state["model_config"], {"refiner": None})
        self.assertEqual(model.weight.item(), 2.0)
        self.assertEqual(model.weight.device.type, "cpu")

    def test_focal_shift_solver_recovers_analytic_camera_on_device(self):
        uv = _normalized_view_plane_uv(32, 48, torch.device("cpu"))
        expected_focal, expected_shift = 0.83, 0.17
        depth = 2.0 + 0.003 * torch.arange(32 * 48).reshape(32, 48)
        points = torch.cat([
            uv * expected_focal * depth[..., None], depth[..., None],
        ], dim=-1)[None]
        points[..., 2] -= expected_shift
        focal, shift = recover_focal_shift_npu(
            points, downsample_size=(32, 48),
        )
        self.assertEqual(focal.device, points.device)
        self.assertEqual(shift.device, points.device)
        torch.testing.assert_close(focal, torch.tensor(expected_focal), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(shift, torch.tensor(expected_shift), atol=1e-5, rtol=1e-5)

    def test_focal_shift_solver_honors_mask_and_nearest_downsampling(self):
        uv = _normalized_view_plane_uv(32, 48, torch.device("cpu"))
        expected_focal, expected_shift = 1.2, -0.1
        depth = 2.0 + 0.002 * torch.arange(32 * 48).reshape(32, 48)
        points = torch.cat([
            uv * expected_focal * depth[..., None], depth[..., None],
        ], dim=-1)[None]
        points[..., 2] -= expected_shift
        points[:, :8] = 1e6
        mask = torch.ones((32, 48), dtype=torch.bool)
        mask[:8] = False
        focal, shift = recover_focal_shift_npu(points, mask=mask)
        torch.testing.assert_close(focal, torch.tensor(expected_focal), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(shift, torch.tensor(expected_shift), atol=1e-5, rtol=1e-5)

    def test_camera_embedding_stays_on_input_device(self):
        intrinsic = torch.tensor([[[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]]])
        w2c = torch.eye(4)[None]
        embedding = get_camera_embedding(intrinsic, w2c, 1, 2, 2)
        self.assertEqual(embedding.device, intrinsic.device)
        self.assertEqual(tuple(embedding.shape), (1, 6, 1, 2, 2))
        self.assertTrue(torch.isfinite(embedding).all())
    def test_subject_geometry_matches_numpy(self):
        import numpy as np
        from src.subject_camera import aim_cameras_at_subject as numpy_aim
        from src.subject_camera import lift_subject_to_world as numpy_lift

        depth = torch.arange(20 * 30, dtype=torch.float32).reshape(20, 30) / 100 + 1
        intrinsic = torch.tensor([[30., 0., 15.], [0., 30., 10.], [0., 0., 1.]])
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor([1., 2., 3.])
        actual = lift_subject_to_world((24, 6), depth, intrinsic, c2w, (30, 20))
        expected = numpy_lift((24, 6), depth.numpy(), intrinsic.numpy(), c2w.numpy(), (30, 20))
        for key in ("source_pixel", "resized_pixel", "depth", "camera_point", "world_point"):
            torch.testing.assert_close(actual[key], torch.as_tensor(expected[key], dtype=torch.float32), atol=1e-5, rtol=1e-5)

        poses = c2w[None].repeat(5, 1, 1)
        poses[:, :3, 3] += torch.arange(5)[:, None] * torch.tensor([0.1, 0.2, -0.3])
        target = torch.tensor([2., 3., 5.])
        actual = aim_cameras_at_subject(poses, target, ramp_fraction=0.5)
        expected = numpy_aim(poses.numpy(), target.numpy(), ramp_fraction=0.5)
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=torch.float32), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
