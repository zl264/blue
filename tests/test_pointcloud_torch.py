"""CPU geometry tests for the pure-PyTorch NPU point-rendering fallback."""
import unittest

import torch

from src.pointcloud_torch import render_points_torch


class TorchPointRendererTest(unittest.TestCase):
    def setUp(self):
        self.k = torch.tensor([[[3., 0., 1.5], [0., 3., 1.5], [0., 0., 1.]]])
        self.w2c = torch.eye(4)[None]

    def test_single_point_color_and_mask(self):
        points = torch.tensor([[0., 0., 2.]])
        colors = torch.tensor([[0.25, -0.5, 0.75]])
        image, mask = render_points_torch(
            points, colors, self.k, self.w2c, 3, 3, splat_radius=0, chunk_size=1,
        )
        self.assertEqual(image.shape, (1, 3, 3, 3))
        self.assertEqual(mask.shape, (1, 1, 3, 3))
        torch.testing.assert_close(image[0, :, 1, 1], colors[0])
        self.assertEqual(mask.sum().item(), 8.0)
        self.assertEqual(mask[0, 0, 1, 1].item(), 0.0)

    def test_nearest_point_wins_depth_test(self):
        points = torch.tensor([[0., 0., 4.], [0., 0., 2.]])
        colors = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        image, mask = render_points_torch(
            points, colors, self.k, self.w2c, 3, 3, splat_radius=0, chunk_size=1,
        )
        torch.testing.assert_close(image[0, :, 1, 1], colors[1])
        self.assertEqual(mask.sum().item(), 8.0)

    def test_translation_and_intrinsics_control_projection(self):
        points = torch.tensor([[0., 0., 2.]])
        colors = torch.ones(1, 3)
        w2c = self.w2c.clone()
        w2c[0, 0, 3] = 0.6666667  # camera x=-2/3, projects to pixel u=2
        image, _ = render_points_torch(
            points, colors, self.k, w2c, 3, 3, splat_radius=0,
        )
        torch.testing.assert_close(image[0, :, 1, 2], colors[0])

    def test_out_of_view_and_behind_camera_are_missing(self):
        points = torch.tensor([[0., 0., -2.], [20., 0., 2.]])
        colors = torch.ones(2, 3)
        image, mask = render_points_torch(
            points, colors, self.k, self.w2c, 3, 3, splat_radius=0,
        )
        self.assertTrue(torch.all(mask == 1))
        self.assertTrue(torch.all(image == 0))

    def test_splat_footprint_and_chunking(self):
        points = torch.tensor([[0., 0., 2.]])
        colors = torch.ones(1, 3)
        image, mask = render_points_torch(
            points, colors, self.k, self.w2c, 3, 3, splat_radius=1, chunk_size=1,
        )
        self.assertEqual(mask.sum().item(), 0.0)
        self.assertTrue(torch.isfinite(image).all())


if __name__ == "__main__":
    unittest.main()
