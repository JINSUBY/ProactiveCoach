# Adapted from the training implementation. See LICENSE.
import torch
from typing import Dict, Any, List, Tuple
from .positions import ROPE_INDEX_FN

def find_assistant_spans(input_ids_1d: List[int], tokenizer) -> List[Tuple[int, int]]:
    assistant_id, im_end_id = tokenizer.convert_tokens_to_ids(['assistant', '<|im_end|>'])
    spans: List[Tuple[int, int]] = []
    L = len(input_ids_1d)
    pos = 0
    while pos < L:
        if input_ids_1d[pos] == assistant_id:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_1d[ans_end] != im_end_id:
                ans_end += 1
            if ans_end < L:
                spans.append((ans_start, ans_end + 2))
                pos = ans_end
        pos += 1
    return spans

def compute_position_ids(processor_output: Dict[str, Any], processor, model_type: str) -> torch.Tensor:
    input_ids = processor_output['input_ids']
    if 'image_grid_thw' in processor_output:
        image_grid_thw = processor_output['image_grid_thw']
        if not isinstance(image_grid_thw, (list, tuple)):
            image_grid_thw = [image_grid_thw]
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
    else:
        image_grid_thw = None
    if 'video_grid_thw' in processor_output:
        video_grid_thw = processor_output['video_grid_thw']
        if not isinstance(video_grid_thw, (list, tuple)):
            video_grid_thw = [video_grid_thw]
        video_grid_thw = torch.cat(video_grid_thw, dim=0)
        second_per_grid_ts = [processor_output.pop('video_chunk_size', 1) * processor.video_processor.temporal_patch_size / processor.video_processor.fps] * len(video_grid_thw)
    else:
        video_grid_thw = None
        second_per_grid_ts = None
    merge_size = getattr(processor.image_processor, 'merge_size', 2)
    rope_fn = ROPE_INDEX_FN[model_type]
    position_ids, _ = rope_fn(merge_size, input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts)
    return position_ids
