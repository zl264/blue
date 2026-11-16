"""Lightweight tests for subject geometry; requires only NumPy and unittest."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.subject_camera import (
    aim_cameras_at_subject, lift_subject_to_world, look_at_rotation,
    resize_subject_pixel,
)


class SubjectGeometryTest(unittest.TestCase):
    def test_resize_and_backprojection(self):
        resized = resize_subject_pixel((960, 540), (1920, 1080), (768, 480))
        np.testing.assert_allclose(resized, [383.7, 215.7])
        source_k = np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]])
        target_k = np.array([[400., 0., 384.], [0., 400., 240.], [0., 0., 1.]])
        depth = np.full((480, 768), 2.)
        subject = lift_subject_to_world((960, 540), depth, target_k, np.eye(4),
                                        (1920, 1080), sample_radius=0)
        np.testing.assert_allclose(subject["camera_point"], [-.00225, -.066, 2.])
        # An override is already in rendered z units; resize never scales it.
        overridden = lift_subject_to_world((960, 540), depth, target_k, np.eye(4),
                                           (1920, 1080), depth_override=3.)
        self.assertEqual(overridden["depth"], 3.)
        np.testing.assert_allclose(overridden["camera_point"], 1.5*subject["camera_point"])

    def test_off_center_lift_uses_world_rotation(self):
        depth = np.full((20, 30), 4.)
        k = np.array([[30., 0., 15.], [0., 30., 10.], [0., 0., 1.]])
        c2w = np.eye(4)
        c2w[:3, :3] = look_at_rotation((0., 0., -4.), (0., 0., 0.))
        c2w[:3, 3] = [1., 2., 3.]
        subject = lift_subject_to_world((24, 6), depth, k, c2w, (30, 20), sample_radius=0)
        expected = c2w[:3, :3] @ np.array([3.6, -1.6, 4.]) + c2w[:3, 3]
        np.testing.assert_allclose(subject["world_point"], expected, atol=1e-6)

    def test_look_at_projects_subject_to_principal_point(self):
        target = np.array([2., -1., 4.])
        for position in [np.zeros(3), np.array([3., 0., 1.]), np.array([2., 0., 4.])]:
            rotation = look_at_rotation(position, target, world_up=(0., -1., 0.))
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            relative = rotation.T @ (target-position)
            self.assertAlmostEqual(relative[0], 0., places=6)
            self.assertAlmostEqual(relative[1], 0., places=6)
            self.assertGreater(relative[2], 0.)
        # Default image-up is parallel to this view, but a fallback is finite.
        rotation = look_at_rotation(np.zeros(3), (0., -2., 0.))
        self.assertTrue(np.isfinite(rotation).all())

    def test_aim_preserves_first_frame_and_handles_off_center_subject(self):
        source = np.eye(4)
        source[:3, :3] = look_at_rotation((0., 0., -4.), (0., 0., 0.))
        source[:3, 3] = [0., 0., -4.]
        target = np.array([.5, .25, 0.])
        poses = np.repeat(source[None], 5, axis=0)
        positions = np.array([[0., 0., -4.], [1., 0., -4.], [0., 1., -4.],
                              [0., 0., -5.], [-1., 0., -4.]])
        poses[:, :3, 3] = positions
        aimed = aim_cameras_at_subject(poses, target, ramp_fraction=0.)
        np.testing.assert_allclose(aimed[0], source, atol=1e-7)
        for pose, position in zip(aimed[1:], positions[1:]):
            relative = pose[:3, :3].T @ (target-position)
            np.testing.assert_allclose(relative[:2], 0., atol=1e-6)
            self.assertGreater(relative[2], 0.)
        self.assertTrue(np.isfinite(aimed).all())
        # The same subject remains centered even after a gradual first ramp.
        ramped = aim_cameras_at_subject(poses, target, ramp_fraction=.5)
        np.testing.assert_allclose(ramped[0], source, atol=1e-7)
        self.assertFalse(np.allclose(ramped[1], aimed[1]))
        np.testing.assert_allclose(ramped[-1], aimed[-1], atol=1e-6)

    def test_invalid_inputs_fail_instead_of_generating_nan(self):
        depth = np.ones((10, 10))
        with self.assertRaisesRegex(ValueError, "outside"):
            resize_subject_pixel((10, 0), (10, 10), (10, 10))
        with self.assertRaisesRegex(ValueError, "positive depth"):
            lift_subject_to_world((5, 5), depth, np.eye(3), np.eye(4), (10, 10), depth_override=-1)
        with self.assertRaisesRegex(ValueError, "coincides"):
            look_at_rotation(np.zeros(3), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "finite positive depth"):
            lift_subject_to_world((5, 5), depth*np.nan, np.eye(3), np.eye(4), (10, 10))


if __name__ == "__main__":
    unittest.main()
