# Adapted from the training implementation. See LICENSE.
import os
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask

def generate_video_sliding_window_mask_mod(video_mask, attention_mask, window_size_n, response_video_mask=None, hidden_watch_mask=None):
    B_limit = video_mask.shape[0]
    L_limit = video_mask.shape[1]
    has_response_mask = response_video_mask is not None
    if not has_response_mask:
        response_video_mask = torch.zeros_like(video_mask)
    has_hidden_watch = hidden_watch_mask is not None
    if not has_hidden_watch:
        hidden_watch_mask = torch.zeros_like(video_mask)
    shifted = F.pad(video_mask[:, :-1], (1, 0), value=False)
    shifted = shifted.contiguous()
    block_starts = video_mask & ~shifted
    block_ids = block_starts.long().cumsum(dim=-1)

    def sliding_window_mod(b, h, q_idx, kv_idx):
        b_c = torch.clamp(b, 0, B_limit - 1)
        q_idx_c = torch.clamp(q_idx, 0, L_limit - 1)
        kv_idx_c = torch.clamp(kv_idx, 0, L_limit - 1)
        in_bounds = (b < B_limit) & (q_idx < L_limit) & (kv_idx < L_limit)
        q_is_valid = attention_mask[b_c, q_idx_c] > 0
        k_is_valid = attention_mask[b_c, kv_idx_c] > 0
        is_valid_pair = in_bounds & q_is_valid & k_is_valid
        is_causal = q_idx_c >= kv_idx_c
        k_is_video = video_mask[b_c, kv_idx_c]
        q_block = block_ids[b_c, q_idx_c]
        k_block = block_ids[b_c, kv_idx_c]
        diff = q_block - k_block
        k_is_response_video = response_video_mask[b_c, kv_idx_c]
        is_in_window = ~k_is_video | (diff < window_size_n) | k_is_response_video
        k_is_hidden_watch = hidden_watch_mask[b_c, kv_idx_c]
        watch_ok = ~k_is_hidden_watch | (q_block == k_block)
        return is_valid_pair & is_causal & is_in_window & watch_ok
    return sliding_window_mod

def create_mask(mask_mod, training: bool, B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
    if torch.compiler.is_compiling():
        create_block_mask_func = create_block_mask
    else:
        create_block_mask_func = torch.compile(create_block_mask)
    tile = int(os.environ.get('THINKSTREAM_MASK_Q_TILE', '0'))
    if tile and Q_LEN > tile:
        if tile % 128:
            raise ValueError('THINKSTREAM_MASK_Q_TILE must be a multiple of 128')
        parts = []
        for start in range(0, Q_LEN, tile):
            offset = torch.tensor(start, device=device)

            def tile_mod(b, h, q, k):
                return mask_mod(b, h, q + offset, k)
            parts.append(create_block_mask_func(tile_mod, B=B, H=H, Q_LEN=min(tile, Q_LEN - start), KV_LEN=KV_LEN, device=device))
        return BlockMask.from_kv_blocks(torch.cat([p.kv_num_blocks for p in parts], dim=-1), torch.cat([p.kv_indices for p in parts], dim=-2), torch.cat([p.full_kv_num_blocks for p in parts], dim=-1), torch.cat([p.full_kv_indices for p in parts], dim=-2), BLOCK_SIZE=parts[0].BLOCK_SIZE, mask_mod=mask_mod, seq_lengths=(Q_LEN, KV_LEN))
    return create_block_mask_func(mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device)

class _BaseCompiledSingleton:

    def __init_subclass__(cls, *, target_fn, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._is_compiled = False
        cls._compiled_fn = None
        cls._target_fn = staticmethod(target_fn)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self, training=True):
        if not self._is_compiled or training != self.training:
            self.training = training
            if self._target_fn is None:
                raise ValueError(f'Class {self.__class__.__name__} has no target_fn defined.')
            self._compiled_fn = torch.compile(self._target_fn)
            self._is_compiled = True

    def __call__(self):
        if self._compiled_fn is None:
            raise RuntimeError(f'{self._target_fn.__name__} is not compiled.')
        return self._compiled_fn

class WrappedFlexAttention(_BaseCompiledSingleton, target_fn=flex_attention):
    pass

def flex_attention_forward(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args, video_block_mask: BlockMask, **kwargs):
    if torch.compiler.is_compiling():
        flex_attn_func = flex_attention
    else:
        flex_attn_func = WrappedFlexAttention(module.training)()
    attn_output = flex_attn_func(query, key, value, block_mask=video_block_mask, enable_gqa=True)
    return (attn_output.transpose(1, 2).contiguous(), None)

def register_streaming_attention():
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register('streaming_attention', flex_attention_forward)
