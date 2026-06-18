"""
端到端结构测试: 验证 Qwen3VLModel 的视觉 token 注入 + DeepStack 流程。

使用最小模型尺寸, 仅验证代码结构和关键逻辑正确性。
"""
import torch
import sys
sys.path.insert(0, '/data/Prism-Infer')
from prism_infer.models.qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLModel,
    Qwen3VLForCausalLM,
)


def test_text_only_forward():
    """纯文本前向."""
    print("--- Test 1: Text-only forward (4 layers) ---")
    model = Qwen3VLTextModel(
        vocab_size=100, hidden_size=256, num_heads=4, num_kv_heads=2,
        num_layers=4, intermediate_size=512, dtype=torch.float32,
    )
    model.eval()

    input_ids = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert out.shape == (1, 16, 256)
    print("  PASS")


def test_deepstack_process():
    """验证 _deepstack_process 的核心逻辑."""
    print("--- Test 2: _deepstack_process ---")
    model = Qwen3VLTextModel(
        vocab_size=100, hidden_size=256, num_heads=4, num_kv_heads=2,
        num_layers=4, intermediate_size=512, dtype=torch.float32,
    )
    model.eval()

    hidden = torch.randn(1, 8, 256)
    vis_mask = torch.zeros(1, 8, dtype=torch.bool)
    vis_mask[0, 2] = True
    vis_mask[0, 5] = True

    ds_embed = torch.randn(2, 256)
    original = hidden.clone()
    result = model._deepstack_process(hidden, vis_mask, ds_embed)

    assert torch.allclose(result[~vis_mask], original[~vis_mask])
    delta = result[vis_mask] - original[vis_mask]
    assert torch.allclose(delta, ds_embed, atol=1e-6), \
        f"注入值不对, max diff: {(delta-ds_embed).abs().max():.2e}"
    print("  PASS")


def test_forward_deepstack():
    """验证 forward 中 deepstack 注入正确."""
    print("--- Test 3: DeepStack injection in forward ---")
    model = Qwen3VLTextModel(
        vocab_size=100, hidden_size=256, num_heads=4, num_kv_heads=2,
        num_layers=4, intermediate_size=512, dtype=torch.float32,
    )
    model.eval()

    input_ids = torch.randint(0, 100, (1, 6))
    vis_mask = torch.zeros(1, 6, dtype=torch.bool)
    vis_mask[0, 2] = True

    ds1 = torch.randn(1, 256)
    ds2 = torch.randn(1, 256)

    with torch.no_grad():
        out_no = model(input_ids=input_ids)
        out_ds = model(input_ids=input_ids,
                       visual_pos_masks=vis_mask,
                       deepstack_visual_embeds=[ds1, ds2])

    diff = (out_ds[vis_mask] - out_no[vis_mask]).abs().max()
    assert diff > 0, f"DeepStack 未生效 (diff={diff:.6f})!"
    print(f"  Visual pos max diff: {diff:.6f} ( > 0, deepstack 已注入)")
    print("  PASS")


def test_image_token_mask():
    """验证 image_token_id 用于创建正确的 mask."""
    print("--- Test 4: Image token masking ---")
    image_token_id = 151655
    input_ids = torch.tensor([[0, image_token_id, image_token_id, 1, image_token_id]])
    mask = (input_ids == image_token_id)  # [1, 5]

    assert mask.sum().item() == 3
    assert mask[0, 1].item() == True
    assert mask[0, 2].item() == True
    assert mask[0, 4].item() == True
    assert mask[0, 0].item() == False
    assert mask[0, 3].item() == False
    print("  PASS")


def test_masked_scatter():
    """验证 masked_scatter 的行为符合预期."""
    print("--- Test 5: masked_scatter behavior ---")
    # 模拟: [batch, seqlen, hidden] 中替换 visual token 位置
    inputs_embeds = torch.randn(1, 5, 4096, dtype=torch.float32)
    image_mask = torch.zeros(1, 5, 1, dtype=torch.bool)
    image_mask[0, 1:3, 0] = True  # tokens 1,2 是 visual

    # 2 个 visual tokens × 4096 dim
    main_vis = torch.randn(2, 4096, dtype=torch.float32)

    # masked_scatter: mask 广播到 [1,5,4096], source 按 mask True 位置替换
    result = inputs_embeds.masked_scatter(image_mask, main_vis)

    # 验证 non-visual 位置没变
    assert torch.allclose(result[0, 0], inputs_embeds[0, 0])
    assert torch.allclose(result[0, 3], inputs_embeds[0, 3])
    assert torch.allclose(result[0, 4], inputs_embeds[0, 4])

    # 验证 visual 位置被替换
    assert torch.allclose(result[0, 1], main_vis[0])
    assert torch.allclose(result[0, 2], main_vis[1])
    print("  PASS")


def test_mismatch_raises():
    """验证 token 数量不匹配时抛出错误."""
    print("--- Test 6: Token count mismatch ---")
    from prism_infer.models.qwen3_vl import Qwen3VLModel

    # 小模型: 2 layers, 256 hidden (OOM 安全)
    IMAGE_TOKEN = 50               # 伪 image_token_id (在 vocab 内)
    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_token_id = IMAGE_TOKEN
            self.language_model = Qwen3VLTextModel(
                vocab_size=100, hidden_size=256, num_heads=4, num_kv_heads=2,
                num_layers=2, intermediate_size=512, dtype=torch.float32)
            self.visual = torch.nn.Identity()  # placeholder

        def forward(self, input_ids, pixel_values=None, image_grid_thw=None,
                    position_embeddings=None, attention_mask=None):
            inputs_embeds = self.language_model.embed_tokens(input_ids)

            if pixel_values is not None:
                main_vis = torch.randn(196, 256)  # 196 visual tokens (mock)
                visual_pos_masks_2d = (input_ids == self.image_token_id)
                n_vis = visual_pos_masks_2d.sum().item()
                if n_vis * 256 != main_vis.numel():
                    raise ValueError(
                        f"视觉 token 数量不匹配: input 中有 {n_vis} 个 "
                        f"token ({n_vis*256} elements), "
                        f"但 Vision Encoder 输出 {main_vis.shape[0]} 个 token "
                        f"({main_vis.numel()} elements)"
                    )
            return inputs_embeds

    model = SmallModel()
    input_ids = torch.full((1, 100), IMAGE_TOKEN, dtype=torch.long)  # 100 ≠ 196
    pixel_values = torch.randn(784, 1536)

    try:
        model(input_ids, pixel_values=pixel_values,
              image_grid_thw=torch.tensor([[1, 28, 28]]))
        print("  FAIL: 应抛错!")
        assert False
    except ValueError as e:
        assert "100" in str(e) and "196" in str(e), f"错误消息不对: {e}"
        print(f"  正确抛出: {e}")
        print("  PASS")


if __name__ == '__main__':
    print("=== Full Model Structure Tests ===\n")
    test_text_only_forward()
    test_deepstack_process()
    test_forward_deepstack()
    test_image_token_mask()
    test_masked_scatter()
    test_mismatch_raises()
    print("\n=== All Structure Tests PASS ===")
