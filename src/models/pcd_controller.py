import os
from typing import Any, Dict, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from huggingface_hub import hf_hub_download
from src.accelerator import autocast_for
from src.models.controlnet import (
    DeviceWanRotaryPosEmbed, WanXControlNet, configure_device_attention, zero_module,
)

if os.getenv("WORLD_SIZE", "1") != "1":
    from xfuser.core.distributed import get_sequence_parallel_rank, get_sp_group

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MaskCamEmbed(nn.Module):
    def __init__(self, controlnet_cfg) -> None:
        super().__init__()

        # padding bug fixed
        if controlnet_cfg.get("interp", False):
            self.mask_padding = [0, 0, 0, 0, 3, 3]  # 左右上下前后, I2V-interp，首尾帧
        else:
            self.mask_padding = [0, 0, 0, 0, 3, 0]  # 左右上下前后, I2V
        add_channels = controlnet_cfg.get("add_channels", 1)
        mid_channels = controlnet_cfg.get("mid_channels", 64)
        self.mask_proj = nn.Sequential(nn.Conv3d(add_channels, mid_channels, kernel_size=(4, 8, 8), stride=(4, 8, 8)),
                                       nn.GroupNorm(mid_channels // 8, mid_channels), nn.SiLU())
        self.mask_zero_proj = zero_module(nn.Conv3d(mid_channels, controlnet_cfg.conv_out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)))

    def forward(self, add_inputs: torch.Tensor):
        # render_mask.shape [b,c,f,h,w]
        warp_add_pad = F.pad(add_inputs, self.mask_padding, mode="constant", value=0)
        add_embeds = self.mask_proj(warp_add_pad)  # [B,C,F,H,W]
        add_embeds = self.mask_zero_proj(add_embeds)
        add_embeds = einops.rearrange(add_embeds, "b c f h w -> b (f h w) c")

        return add_embeds


class PCDController(WanTransformer3DModel):
    r"""
    A Transformer model for video-like data used in the Wan model.
    """

    def __init__(
            self,
            patch_size: Tuple[int] = (1, 2, 2),
            num_attention_heads: int = 40,
            attention_head_dim: int = 128,
            in_channels: int = 16,
            out_channels: int = 16,
            text_dim: int = 4096,
            freq_dim: int = 256,
            ffn_dim: int = 13824,
            num_layers: int = 40,
            cross_attn_norm: bool = True,
            qk_norm: Optional[str] = "rms_norm_across_heads",
            eps: float = 1e-6,
            image_dim: Optional[int] = None,
            added_kv_proj_dim: Optional[int] = None,
            rope_max_seq_len: int = 1024,
            controlnet_cfg=None
    ) -> None:
        super().__init__(patch_size=patch_size,
                         num_attention_heads=num_attention_heads,
                         attention_head_dim=attention_head_dim,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         text_dim=text_dim,
                         freq_dim=freq_dim,
                         ffn_dim=ffn_dim,
                         num_layers=num_layers,
                         cross_attn_norm=cross_attn_norm,
                         qk_norm=qk_norm,
                         eps=eps,
                         image_dim=image_dim,
                         added_kv_proj_dim=added_kv_proj_dim,
                         rope_max_seq_len=rope_max_seq_len)

        self.controlnet_cfg = controlnet_cfg
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.rope_max_seq_len = rope_max_seq_len
        self.sp_size = 1

    @property
    def dtype(self):
        # PEFT can add FP32 LoRA parameters to an otherwise BF16 Wan model.
        # ModelMixin.dtype would then report FP32 and break mixed-precision
        # training/inference, so anchor the dtype on the frozen patch embedding.
        return self.patch_embedding.weight.dtype

    def build_controlnet(self, model_path, logger=None):
        # controlnet
        self.controlnet_patch_embedding = nn.Conv3d(
            self.in_channels, self.controlnet_cfg.conv_out_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        if self.in_channels != 36:
            raise ValueError("Uni3C static training requires Wan I2V in_channels=36 (16 noisy + 20 condition)")
        self.controlnet_mask_embedding = MaskCamEmbed(self.controlnet_cfg)
        self.controlnet = WanXControlNet(self.controlnet_cfg)
        self.controlnet_rope = DeviceWanRotaryPosEmbed(
            self.controlnet_cfg.dim // self.controlnet_cfg.num_heads,
            self.patch_size, self.rope_max_seq_len,
        )
        # from_pretrained casts existing Wan modules to BF16, but modules built
        # afterwards start in FP32. State that contract explicitly for both the
        # original inference path and cached static-scene training.
        self.controlnet_patch_embedding.float()
        self.controlnet_mask_embedding.float()

        if not os.path.exists(model_path):  # download weights from huggingface
            model_path = hf_hub_download(repo_id="ewrfcas/Uni3C", filename=model_path, repo_type="model")
        state_dict = torch.load(model_path, map_location="cpu")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        # print("Missing keys:", missing_keys)
        if logger is not None:
            logger.info(f"Unexpected keys: {unexpected_keys}")
        else:
            print("Unexpected keys:", unexpected_keys)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: Optional[torch.Tensor] = None,
            render_latent=None,
            render_mask=None,
            camera_embedding=None,
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param render_latent: [b,c,f,h,w]
        :param render_mask: [b,1,f,h,w]
        :param camera_embedding: [b,6,f,h,w]
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND and lora_scale != 1.0:
            scale_lora_layers(self, lora_scale)
        try:
            return self._forward(
                hidden_states=hidden_states, timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                render_latent=render_latent, render_mask=render_mask,
                camera_embedding=camera_embedding, return_dict=return_dict,
            )
        finally:
            if USE_PEFT_BACKEND and lora_scale != 1.0:
                unscale_lora_layers(self, lora_scale)

    def _forward(
            self, hidden_states, timestep, encoder_hidden_states,
            encoder_hidden_states_image=None, render_latent=None,
            render_mask=None, camera_embedding=None, return_dict=True,
    ):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        if hidden_states.ndim != 5 or num_channels != self.in_channels:
            raise ValueError(f"hidden_states must be [B,{self.in_channels},F,H,W] for Wan I2V")
        if render_latent is None or render_latent.shape != (batch_size, 16, num_frames, height, width):
            raise ValueError("render_latent must be [B,16,F,latent_H,latent_W]")
        if render_mask is None or render_mask.ndim != 5 or render_mask.shape[0] != batch_size or render_mask.shape[1] != 1:
            raise ValueError("render_mask must be [B,1,F,H,W]")
        if camera_embedding is not None and (
            camera_embedding.ndim != 5 or camera_embedding.shape[1] != 6
        ):
            raise ValueError("camera_embedding must be [B,6,F,H,W]")

        ### process controlnet inputs ###
        # The dynamically built controller starts in FP32. Keep its render/mask
        # inputs FP32 too, then enter the BF16 Wan token path. Uni3C's control
        # embedding consumes the first 20 channels (16 noisy latent + first
        # four I2V condition channels) plus the 16-channel render latent.
        # Thus both patch-embedding Conv3d layers have self.in_channels=36.
        condition = hidden_states[:, :20].float()
        hidden_states = hidden_states.to(self.dtype)
        control_input = torch.cat([condition, render_latent.float()], dim=1)
        controlnet_rotary_emb = self.controlnet_rope(control_input)
        controlnet_inputs = self.controlnet_patch_embedding(control_input)
        controlnet_inputs = controlnet_inputs.flatten(2).transpose(1, 2)

        # additional inputs (mask, camera embedding)
        render_mask = render_mask.float()
        camera_embedding = camera_embedding.float() if camera_embedding is not None else None
        if camera_embedding is not None:
            add_inputs = torch.cat([render_mask, camera_embedding], dim=1)
        else:
            add_inputs = render_mask
        add_inputs = self.controlnet_mask_embedding(add_inputs)
        controlnet_inputs = (controlnet_inputs + add_inputs).to(self.dtype)
        controlnet_rotary_emb = controlnet_rotary_emb.to(hidden_states.device)

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        ### controlnet encoding ###
        if self.sp_size > 1:
            assert controlnet_inputs.shape[1] % self.sp_size == 0
            controlnet_inputs = torch.chunk(controlnet_inputs, self.sp_size, dim=1)[get_sequence_parallel_rank()]
            controlnet_rotary_emb = torch.chunk(controlnet_rotary_emb, self.sp_size, dim=2)[get_sequence_parallel_rank()]

        with autocast_for(hidden_states, dtype=self.dtype, enabled=True):
            controlnet_states = self.controlnet(hidden_states=controlnet_inputs,
                                                temb=temb,
                                                rotary_emb=controlnet_rotary_emb)
        ### controlnet encoding over ###

        ### sp
        if self.sp_size > 1:
            assert hidden_states.shape[1] % self.sp_size == 0
            hidden_states = torch.chunk(hidden_states, self.sp_size, dim=1)[get_sequence_parallel_rank()]
            rotary_emb = torch.chunk(rotary_emb, self.sp_size, dim=2)[get_sequence_parallel_rank()]

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for i, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                # adding control features
                if i < len(controlnet_states):
                    hidden_states += controlnet_states[i]
        else:
            for i, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                # adding control features
                if i < len(controlnet_states):
                    hidden_states += controlnet_states[i]

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if self.sp_size > 1:
            hidden_states = get_sp_group().all_gather(hidden_states, dim=1)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
