# Adapted from the training implementation. See LICENSE.
import os
import torch
GRPO_DIAG: dict = {}
torch._dynamo.config.recompile_limit = int(os.environ.get('THINKSTREAM_RECOMPILE_LIMIT', '64'))
_mbx = os.environ.get('THINKSTREAM_TRITON_MAXBLOCK_X', '')
if _mbx:
    from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
    TRITON_MAX_BLOCK['X'] = int(_mbx)
    import torch._inductor.config as _ic
    _ic.compile_threads = 1
from liger_kernel.transformers.model import loss_utils
LigerForCausalLMLoss = loss_utils.LigerForCausalLMLoss
_TLW_MAX_LEVELS = int(os.environ.get('THINKSTREAM_TLW_MAX_LEVELS', '6'))
_TLW_LOGGED = False

def _merge_weight_levels(levels, counts, max_levels):
    groups = [([v], v, c) for v, c in zip(levels, counts)]
    while len(groups) > max_levels:
        i = min(range(len(groups) - 1), key=lambda j: groups[j + 1][1] / max(groups[j][1], 1e-12))
        (lv_a, w_a, n_a), (lv_b, w_b, n_b) = (groups[i], groups[i + 1])
        n = n_a + n_b
        groups[i:i + 2] = [(lv_a + lv_b, (w_a * n_a + w_b * n_b) / n, n)]
    return groups

def _LigerForCausalLMLoss(*args, **kwargs):
    kwargs.pop('video_block_mask')
    tlw = kwargs.pop('token_loss_weight', None)
    labels = kwargs.get('labels')
    if tlw is None or labels is None or kwargs.get('shift_labels') is not None:
        return LigerForCausalLMLoss(*args, **kwargs)
    ignore = kwargs.get('ignore_index', -100)
    sup = labels[..., 1:] != ignore
    w_sup = tlw[..., 1:][sup]
    if w_sup.numel() == 0:
        return LigerForCausalLMLoss(*args, **kwargs)
    lv, cnt = torch.unique(w_sup, return_counts=True)
    if lv.numel() == 1 and float(lv[0]) == 1.0:
        return LigerForCausalLMLoss(*args, **kwargs)
    groups = _merge_weight_levels(lv.tolist(), cnt.tolist(), _TLW_MAX_LEVELS)
    n_total = float(sum((n for _, _, n in groups)))
    global _TLW_LOGGED
    if not _TLW_LOGGED:
        _TLW_LOGGED = True
        try:
            import torch.distributed as _dist
            rank0 = not (_dist.is_available() and _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            rank0 = True
        if rank0:
            print('[token_loss_weight] ACTIVE — levels(w×n): ' + ' '.join((f'{w:.4g}×{n}' for _, w, n in groups)) + f' | passes={sum((1 for _, w, _ in groups if w != 0.0))} | reduction={('sum' if kwargs.get('num_items_in_batch') is not None else 'mean')}', flush=True)
    by_sum = kwargs.get('num_items_in_batch') is not None
    want_acc = bool(kwargs.get('return_token_accuracy', False))
    loss = None
    acc = None
    for src_levels, w, n in groups:
        if w == 0.0:
            continue
        sel = torch.isin(tlw, torch.tensor(src_levels, dtype=tlw.dtype, device=tlw.device))
        res = LigerForCausalLMLoss(*args, **{**kwargs, 'labels': labels.masked_fill(~sel, ignore)})
        loss_g, _, acc_g = loss_utils.unpack_cross_entropy_result(res)
        contrib = w * loss_g if by_sum else w * n / n_total * loss_g
        loss = contrib if loss is None else loss + contrib
        if want_acc and acc_g is not None:
            a = acc_g * (n / n_total)
            acc = a if acc is None else acc + a
    if loss is None:
        loss = LigerForCausalLMLoss(*args, **kwargs) * 0.0
    if want_acc:
        return loss_utils.CrossEntropyOutput(loss=loss, token_accuracy=acc)
    return loss
loss_utils.LigerForCausalLMLoss = _LigerForCausalLMLoss
from transformers.utils import can_return_tuple
from liger_kernel.transformers.model import qwen2_5_vl
from liger_kernel.transformers.model import qwen3_vl
lce_forward_qwen2_5_vl = qwen2_5_vl.lce_forward
lce_forward_qwen3_vl = qwen3_vl.lce_forward
from .attention import create_mask, generate_video_sliding_window_mask_mod
DEFAULT_VIDEO_FLEX_WINDOW_SIZE = 5

def build_video_block_mask(model, video_mask, attention_mask, response_video_mask=None, hidden_watch_mask=None):
    if video_mask is None:
        return None
    assert attention_mask is not None
    window_size_n = getattr(model.config, 'video_flex_window_size', DEFAULT_VIDEO_FLEX_WINDOW_SIZE)
    B, L = video_mask.shape
    assert video_mask.shape == attention_mask.shape
    mask_mod = generate_video_sliding_window_mask_mod(video_mask.contiguous(), attention_mask.contiguous(), window_size_n, response_video_mask=response_video_mask.contiguous() if response_video_mask is not None else None, hidden_watch_mask=hidden_watch_mask.contiguous() if hidden_watch_mask is not None else None)
    return create_mask(mask_mod, model.training, B=B, H=None, Q_LEN=L, KV_LEN=L, device=model.device)

@can_return_tuple
def _lce_forward_qwen2_5_vl(self, *args, attention_mask=None, video_mask=None, response_video_mask=None, hidden_watch_mask=None, **kwargs):
    video_block_mask = build_video_block_mask(self, video_mask, attention_mask, response_video_mask, hidden_watch_mask)
    return lce_forward_qwen2_5_vl(self, *args, attention_mask=attention_mask, video_block_mask=video_block_mask, **kwargs)

@can_return_tuple
def _lce_forward_qwen3_vl(self, *args, attention_mask=None, video_mask=None, response_video_mask=None, hidden_watch_mask=None, **kwargs):
    video_block_mask = None
    if self.config.text_config._attn_implementation == 'streaming_attention':
        video_block_mask = build_video_block_mask(self, video_mask, attention_mask, response_video_mask, hidden_watch_mask)
    return lce_forward_qwen3_vl(self, *args, attention_mask=attention_mask, video_block_mask=video_block_mask, **kwargs)
qwen2_5_vl.lce_forward = _lce_forward_qwen2_5_vl
qwen3_vl.lce_forward = _lce_forward_qwen3_vl
print('Successfully Patched!')
try:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as _m35
    _HAS_QWEN35 = True
except ImportError:
    _HAS_QWEN35 = False
if _HAS_QWEN35:
    from types import MethodType as _MethodType
    from liger_kernel.transformers import monkey_patch as _liger_mp
    from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result as _unpack_ce_result
    from liger_kernel.transformers.model.output_classes import LigerQwen3VLCausalLMOutputWithPast as _LigerQ35Output
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS as _MASK_FNS, flash_attention_mask as _flash_mask_fn
    if 'streaming_attention' not in _MASK_FNS._global_mapping:
        _MASK_FNS.register('streaming_attention', _flash_mask_fn)

    @can_return_tuple
    def lce_forward_qwen3_5(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None, logits_to_keep=0, skip_logits=None, use_cache=None, cache_position=None, video_mask=None, response_video_mask=None, hidden_watch_mask=None, video_block_mask=None, second_per_grid_ts=None, rope_deltas=None, **kwargs):
        if os.environ.get('THINKSTREAM_Q35_WINDOW', '') == '1' and video_mask is not None:
            kwargs['video_block_mask'] = build_video_block_mask(self, video_mask, attention_mask, response_video_mask, hidden_watch_mask)
        loss_kwargs = {}
        for _k in ('token_loss_weight', 'ce_weight', 'num_items_in_batch', 'shift_labels'):
            if _k in kwargs:
                loss_kwargs[_k] = kwargs.pop(_k)
        if cache_position is not None:
            kwargs['cache_position'] = cache_position
        if use_cache is not None:
            kwargs['use_cache'] = use_cache
        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, mm_token_type_ids=mm_token_type_ids, **kwargs)
        hidden_states = outputs[0]
        shift_labels = loss_kwargs.pop('shift_labels', None)
        loss = None
        logits = None
        token_accuracy = None
        if skip_logits and labels is None and (shift_labels is None):
            raise ValueError('skip_logits is True, but labels and shift_labels are None')
        if skip_logits is None:
            skip_logits = self.training and (labels is not None or shift_labels is not None)
        if skip_logits:
            result = loss_utils.LigerForCausalLMLoss(hidden_states=hidden_states, lm_head_weight=self.lm_head.weight, labels=labels, shift_labels=shift_labels, hidden_size=self.config.text_config.hidden_size, video_block_mask=None, **loss_kwargs)
            loss, _, token_accuracy = _unpack_ce_result(result)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
        return _LigerQ35Output(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, rope_deltas=outputs.rope_deltas, token_accuracy=token_accuracy)

    def apply_liger_kernel_to_qwen3_5(rope: bool=True, cross_entropy: bool=False, fused_linear_cross_entropy: bool=True, rms_norm: bool=True, swiglu: bool=True, model=None, **kwargs) -> None:
        if not fused_linear_cross_entropy:
            return
        _m35.Qwen3_5ForConditionalGeneration.forward = lce_forward_qwen3_5
        if model is not None and isinstance(model, _m35.Qwen3_5ForConditionalGeneration):
            model.forward = _MethodType(lce_forward_qwen3_5, model)
    _liger_mp.MODEL_TYPE_TO_APPLY_LIGER_FN['qwen3_5'] = apply_liger_kernel_to_qwen3_5
    print('[patch] qwen3_5 liger lce_forward registered')
