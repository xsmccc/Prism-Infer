"""Qwen3-VL 文本侧 M-RoPE position ids 构造。

本模块自实现 Qwen3-VL 图文 prefill 的 3D position ids 和 rope_delta。
参考 HF 4.57.1:
`transformers/models/qwen3_vl/modeling_qwen3_vl.py:916-1033`。

P3.2 扩展为同时支持 image/video span。视频语义与 HF 保持一致:
先按帧展开 `video_grid_thw`，再把 T 置为 1；时间信息由
processor 展开的 timestamp 文本 token 承载。
"""

from __future__ import annotations

import torch


def get_qwen3_vl_rope_index(
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 Qwen3-VL 文本模型使用的 3D position ids 和 rope_delta。

    input_ids: [batch, seqlen]
    image_grid_thw: [num_images, 3] or None
    video_grid_thw: [num_videos, 3] or None
    attention_mask: [batch, seqlen] or None
    返回:
      position_ids: [3, batch, seqlen]
      rope_delta: [batch, 1]
    """

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seqlen], got {list(input_ids.shape)}")
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask shape must match input_ids, "
            f"got {list(attention_mask.shape)} vs {list(input_ids.shape)}"
        )

    if image_grid_thw is None and video_grid_thw is None:
        return _get_text_rope_index(input_ids, attention_mask)

    if image_grid_thw is not None and (
        image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3
    ):
        raise ValueError(
            f"image_grid_thw must have shape [num_images, 3], got {list(image_grid_thw.shape)}"
        )
    if video_grid_thw is not None and (
        video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3
    ):
        raise ValueError(
            f"video_grid_thw must have shape [num_videos, 3], got {list(video_grid_thw.shape)}"
        )

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw,
            video_grid_thw[:, 0],
            dim=0,
        ).clone()
        video_grid_thw[:, 0] = 1

    return _get_visual_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        spatial_merge_size=spatial_merge_size,
    )


def _get_text_rope_index(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """纯文本 position ids，与 HF text-only 分支一致。"""

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        rope_delta = max_position_ids + 1 - attention_mask.shape[-1]
        return position_ids.to(dtype=input_ids.dtype), rope_delta.to(dtype=input_ids.dtype)

    position_ids = (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .view(1, 1, -1)
        .expand(3, input_ids.shape[0], -1)
    )
    rope_delta = torch.zeros(
        [input_ids.shape[0], 1],
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    return position_ids.to(dtype=input_ids.dtype), rope_delta


def _get_visual_rope_index(
    *,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """图像/视频输入 position ids，与 HF visual 分支逐步对齐。"""

    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)

    position_ids = torch.ones(
        3,
        total_input_ids.shape[0],
        total_input_ids.shape[1],
        dtype=total_input_ids.dtype,
        device=total_input_ids.device,
    )
    rope_deltas: list[torch.Tensor] = []
    image_index = 0
    video_index = 0
    attention_mask = attention_mask.to(total_input_ids.device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(total_input_ids.device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(total_input_ids.device)

    for batch_idx, seq_input_ids in enumerate(total_input_ids):
        active_input_ids = seq_input_ids[attention_mask[batch_idx] == 1]
        vision_start_indices = torch.argwhere(
            active_input_ids == vision_start_token_id
        ).squeeze(1)
        vision_tokens = active_input_ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum().item())
        video_nums = int((vision_tokens == video_token_id).sum().item())
        input_tokens = active_input_ids.tolist()

        llm_pos_ids_list: list[torch.Tensor] = []
        start = 0
        remain_images = image_nums
        remain_videos = video_nums
        for _ in range(image_nums + video_nums):
            if remain_images > 0 and image_token_id in input_tokens:
                image_end = input_tokens.index(image_token_id, start)
            else:
                image_end = len(input_tokens) + 1
            if remain_videos > 0 and video_token_id in input_tokens:
                video_end = input_tokens.index(video_token_id, start)
            else:
                video_end = len(input_tokens) + 1

            if image_end < video_end:
                if image_grid_thw is None or image_index >= image_grid_thw.shape[0]:
                    rows = 0 if image_grid_thw is None else image_grid_thw.shape[0]
                    raise ValueError(
                        f"image_grid_thw has {rows} rows, "
                        f"but input_ids contains at least {image_index + 1} image spans"
                    )
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                visual_end = image_end
            else:
                if video_grid_thw is None or video_index >= video_grid_thw.shape[0]:
                    rows = 0 if video_grid_thw is None else video_grid_thw.shape[0]
                    raise ValueError(
                        f"video_grid_thw has {rows} rows, "
                        f"but input_ids contains at least {video_index + 1} video spans"
                    )
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_videos -= 1
                visual_end = video_end

            llm_grid_t = int(t.item())
            llm_grid_h = int(h.item()) // spatial_merge_size
            llm_grid_w = int(w.item()) // spatial_merge_size
            if min(llm_grid_t, llm_grid_h, llm_grid_w) <= 0:
                raise ValueError(
                    "invalid merged visual grid: "
                    f"raw={[int(t.item()), int(h.item()), int(w.item())]}, "
                    f"merge={spatial_merge_size}"
                )

            text_len = visual_end - start
            start_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                torch.arange(text_len, device=total_input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + start_idx
            )

            # visual token position ids: [3, llm_grid_t * llm_grid_h * llm_grid_w]
            t_index = (
                torch.arange(llm_grid_t, device=total_input_ids.device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, device=total_input_ids.device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=total_input_ids.device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + start_idx
            )
            start = visual_end + llm_grid_t * llm_grid_h * llm_grid_w

        if start < len(input_tokens):
            start_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - start
            llm_pos_ids_list.append(
                torch.arange(text_len, device=total_input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + start_idx
            )

        if not llm_pos_ids_list:
            raise ValueError(
                "visual grid was provided but no image/video spans were found in input_ids"
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        active_mask = attention_mask[batch_idx] == 1
        if llm_positions.shape[1] != int(active_mask.sum().item()):
            raise ValueError(
                "position length mismatch: "
                f"positions={llm_positions.shape[1]}, active_tokens={int(active_mask.sum().item())}"
            )
        position_ids[..., batch_idx, active_mask] = llm_positions.to(position_ids.device)
        rope_deltas.append(llm_positions.max() + 1 - len(total_input_ids[batch_idx]))

    if image_grid_thw is not None and image_index != image_grid_thw.shape[0]:
        raise ValueError(
            f"image_grid_thw has {image_grid_thw.shape[0]} rows, but only {image_index} image spans were used"
        )
    if video_grid_thw is not None and video_index != video_grid_thw.shape[0]:
        raise ValueError(
            f"video_grid_thw has {video_grid_thw.shape[0]} rows, but only {video_index} video spans were used"
        )

    rope_delta = torch.stack(rope_deltas).to(device=input_ids.device, dtype=input_ids.dtype).unsqueeze(1)
    return position_ids, rope_delta


def get_qwen3_vl_rope_index_from_config(
    input_ids: torch.Tensor,
    *,
    config,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 HF config-like 对象读取 token ids 和 merge size 后生成 rope index。"""

    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise ValueError("config does not expose vision_config")
    return get_qwen3_vl_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        image_token_id=int(getattr(config, "image_token_id")),
        video_token_id=int(getattr(config, "video_token_id")),
        vision_start_token_id=int(getattr(config, "vision_start_token_id")),
        spatial_merge_size=int(getattr(vision_config, "spatial_merge_size")),
    )
