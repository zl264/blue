"""Small CPU tests for the static-scene training tensor/adapter contracts."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    move_cache, validate_cache, validate_cache_batch,
)


def synthetic_cache(batch=False):
    nframe, height, width = 5, 16, 16
    shape = (16, 2, 8, 8)
    cache = {
        "target_latent": torch.randn(*shape),
        "condition": torch.randn(20, *shape[1:]),
        "render_latent": torch.randn(*shape),
        "render_mask": torch.randint(0, 2, (1, nframe, height, width)),
        "camera_embedding": torch.randn(6, nframe, height, width),
        "prompt_embeds": torch.randn(8, 16),
        "negative_prompt_embeds": torch.randn(8, 16),
        "image_embeds": torch.randn(257, 32),
    }
    if batch:
        cache = {key: value.unsqueeze(0) for key, value in cache.items()}
    return cache


class StaticTrainingTensorTest(unittest.TestCase):
    def test_dataloader_adds_one_batch_dimension(self):
        cache = synthetic_cache()
        validate_cache(cache)
        loader = DataLoader(TensorDataset(*cache.values()), batch_size=1)
        values = next(iter(loader))
        batch = dict(zip(cache, values))
        batch = move_cache(batch, torch.device("cpu"))
        validate_cache_batch(batch)
        self.assertEqual(batch["target_latent"].shape, (1, 16, 2, 8, 8))
        self.assertEqual(batch["condition"].shape, (1, 20, 2, 8, 8))
        self.assertEqual(batch["camera_embedding"].shape, (1, 6, 5, 16, 16))
        noisy = torch.randn_like(batch["target_latent"])
        model_input = torch.cat([noisy, batch["condition"]], dim=1)
        controller_input = torch.cat([model_input[:, :20], batch["render_latent"]], dim=1)
        self.assertEqual(model_input.shape[1], 36)
        self.assertEqual(controller_input.shape[1], 36)

    def test_old_32_channel_condition_is_rejected(self):
        cache = synthetic_cache()
        cache["condition"] = torch.randn(32, 2, 8, 8)
        with self.assertRaisesRegex(ValueError, "20"):
            validate_cache(cache)

    def test_camera_mask_combination(self):
        batch = synthetic_cache(batch=True)
        combined = torch.cat([batch["render_mask"], batch["camera_embedding"]], dim=1)
        self.assertEqual(combined.shape, (1, 7, 5, 16, 16))

    def test_flow_matching_endpoints(self):
        target = torch.randn(1, 16, 2, 8, 8)
        noise = torch.randn_like(target)
        for sigma in (0.0, 0.5, 1.0):
            noisy = (1 - sigma) * target + sigma * noise
            self.assertTrue(torch.equal(noisy, target * (1 - sigma) + noise * sigma))
        self.assertTrue(torch.equal((noise - target) + target, noise))


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("diffusers", "peft", "safetensors")),
    "PEFT adapter integration tests require diffusers, peft and safetensors",
)
class PeftAdapterRoundTripTest(unittest.TestCase):
    def test_adapter_only_save_load_and_backward(self):
        from diffusers.models import WanTransformer3DModel
        from peft import LoraConfig, get_peft_model

        from src.training import load_lora, save_lora

        config = dict(
            patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
            in_channels=36, out_channels=16, text_dim=16, freq_dim=8,
            ffn_dim=24, num_layers=1, cross_attn_norm=True,
            qk_norm="rms_norm_across_heads", image_dim=32, added_kv_proj_dim=32,
            rope_max_seq_len=64,
        )
        base = WanTransformer3DModel(**config)
        model = get_peft_model(base, LoraConfig(
            r=2, lora_alpha=2, target_modules=["blocks.0.attn1.to_q"],
            init_lora_weights=True,
        ))
        model.train()
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora_" in name
            if "lora_" in name:
                parameter.data.normal_(0, 0.01)
        hidden = torch.randn(1, 36, 1, 4, 4)
        kwargs = dict(
            hidden_states=hidden, timestep=torch.tensor([500.0]),
            encoder_hidden_states=torch.randn(1, 8, 16),
            encoder_hidden_states_image=torch.randn(1, 257, 32),
            return_dict=False,
        )
        loss = model(**kwargs)[0].square().mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for name, parameter in model.named_parameters()
                            if "lora_" in name))
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                            if "lora_" not in name))
        with tempfile.TemporaryDirectory() as directory:
            save_lora(model, directory)
            self.assertTrue((Path(directory) / "adapter_config.json").is_file())
            self.assertTrue((Path(directory) / "adapter_model.safetensors").is_file())
            restored = load_lora(WanTransformer3DModel(**config), directory, trainable=True)
            self.assertIsInstance(restored, WanTransformer3DModel)
            self.assertTrue(any(parameter.requires_grad for parameter in restored.parameters()))
            with torch.no_grad():
                torch.testing.assert_close(model(**kwargs)[0], restored(**kwargs)[0])

    def test_controller_forward_backward_and_adapter_round_trip(self):
        from src.models.pcd_controller import PCDController
        from src.training import add_controlnet_lora, load_lora, save_lora

        control_config = {
            "num_layers": 1, "dim": 24, "ffn_dim": 48, "num_heads": 2,
            "time_embed_dim": 24, "conv_out_dim": 24, "add_channels": 7,
            "mid_channels": 8,
        }
        config = dict(
            patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
            in_channels=36, out_channels=16, text_dim=16, freq_dim=8,
            ffn_dim=48, num_layers=1, cross_attn_norm=True,
            qk_norm="rms_norm_across_heads", image_dim=32, added_kv_proj_dim=32,
            rope_max_seq_len=64, controlnet_cfg=control_config,
        )

        def make_controller():
            from src.models.controlnet import DeviceWanRotaryPosEmbed, WanXControlNet
            from src.models.pcd_controller import MaskCamEmbed

            controller = PCDController(**config)
            controller.controlnet_patch_embedding = torch.nn.Conv3d(
                36, 24, kernel_size=(1, 2, 2), stride=(1, 2, 2),
            )
            controller.controlnet_mask_embedding = MaskCamEmbed(control_config)
            controller.controlnet = WanXControlNet(control_config)
            controller.controlnet_rope = DeviceWanRotaryPosEmbed(12, (1, 2, 2), 64)
            return controller

        model = add_controlnet_lora(make_controller(), rank=2, alpha=2)
        model.train()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data.normal_(0, 0.01)
        kwargs = dict(
            hidden_states=torch.randn(1, 36, 2, 4, 4),
            render_latent=torch.randn(1, 16, 2, 4, 4),
            render_mask=torch.randint(0, 2, (1, 1, 5, 16, 16)).float(),
            camera_embedding=torch.randn(1, 6, 5, 16, 16),
            timestep=torch.tensor([500.0]),
            encoder_hidden_states=torch.randn(1, 8, 16),
            encoder_hidden_states_image=torch.randn(1, 257, 32),
            return_dict=False,
        )
        prediction = model(**kwargs)[0]
        self.assertEqual(prediction.shape, (1, 16, 2, 4, 4))
        prediction.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for name, parameter in model.named_parameters()
                            if "lora_" in name))
        with tempfile.TemporaryDirectory() as directory:
            save_lora(model, directory)
            restored = load_lora(make_controller(), directory, trainable=True)
            # Restore identical frozen test weights; no model download is used.
            base_state = {key: value for key, value in model.state_dict().items()
                          if "lora_" not in key}
            restored.load_state_dict(base_state, strict=False)
            with torch.no_grad():
                output = restored(**kwargs)[0]
                scaled = restored(**kwargs, attention_kwargs={"scale": 0.5})[0]
                self.assertFalse(torch.equal(output, scaled))
                from peft.tuners.tuners_utils import BaseTunerLayer
                for module in restored.modules():
                    if isinstance(module, BaseTunerLayer):
                        self.assertTrue(all(scale == 1.0 for scale in module.scaling.values()))
                torch.testing.assert_close(restored(**kwargs)[0], output)


if __name__ == "__main__":
    unittest.main()
