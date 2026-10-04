"""Sparse streaming SFT loader: one complete sequence per dataset item."""
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .supervision import PROMPT, apply_loss_mask, parse
from .template import CHAT_TEMPLATE
from .processing import find_assistant_spans, compute_position_ids
from .video import load_video_frames


def validate_record(row):
    start, end = float(row['video_start']), float(row['video_end'])
    if not math.isfinite(start + end) or start < 0 or end <= start:
        raise ValueError('Expected a finite nonempty video interval')
    if row.get('video_chunk_size') != 2:
        raise ValueError('This recipe requires video_chunk_size=2 seconds')
    n = round((end - start) / 2)
    if not math.isclose(end - start, n * 2, abs_tol=1e-5) or not 1 <= n <= 540:
        raise ValueError('Expected 1..540 complete two-second chunks')
    turns = row['thoughts']
    if len(turns) != n:
        raise ValueError('Exactly one target is required for every chunk')
    for i, turn in enumerate(turns):
        if not start + 2*i <= float(turn['timestamp']) < start + 2*(i+1):
            raise ValueError(f'Target {i} is outside its original chunk')
        if turn.get('think', ''):
            raise ValueError('The sparse recipe has no think block')
        parse(turn['assistant'])
    queries = row['conversations']
    if not queries or any(q['role'] != 'user' for q in queries):
        raise ValueError('conversations must contain only timestamped user instructions')
    if any(not start <= float(q['timestamp']) < end for q in queries):
        raise ValueError('User instruction lies outside the video interval')
    if not any(float(q['timestamp']) < start + 2 for q in queries):
        raise ValueError('The task goal must be present in the first chunk')
    return n


def build_messages(row, video_path):
    n = validate_record(row)
    start = float(row['video_start'])
    queries = sorted(row['conversations'], key=lambda q: q['timestamp'])
    messages = [{'role': 'system', 'content': PROMPT}]
    for i in range(n):
        lo, hi = start + 2*i, start + 2*(i+1)
        content = [{'type': 'video', 'video': str(video_path),
                    'video_start': lo, 'video_end': hi}]
        content.extend({'type': 'text', 'text': '\n' + q['content']}
                       for q in queries if lo <= q['timestamp'] < hi)
        messages.append({'role': 'user', 'content': content})
        messages.append({'role': 'assistant', 'content': [
            {'type': 'text', 'text': row['thoughts'][i]['assistant']} ]})
    return messages


def encode_record(row, processor, model_type, video_root, max_length=98304,
                  min_pixels=100352, max_pixels=150528):
    n = validate_record(row)
    path = Path(row['video_path'])
    if not path.is_absolute():
        path = Path(video_root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    messages = build_messages(row, path)
    videos, kwargs, metadata = load_video_frames(
        str(path), row['video_start'], row['video_end'], 2*n, 2, n,
        min_pixels=min_pixels, max_pixels=max_pixels,
        processor=processor, model_type=model_type)
    if len(videos) != n or any(len(v) != 2 for v in videos):
        raise ValueError('Decoded frames do not cover every annotated chunk')
    text = processor.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=False,
                                        chat_template=CHAT_TEMPLATE)
    result = dict(processor(text=text, videos=videos, images=None,
                            video_metadata=metadata, do_resize=False,
                            return_tensors='pt', **kwargs))
    ids = result['input_ids']
    if ids.shape[1] > max_length:
        raise ValueError(f'Sequence has {ids.shape[1]} tokens; limit is {max_length}. '
                         'Refusing to truncate training targets.')
    video_id = processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')
    result['video_mask'] = ids == video_id
    # No permanent exception for the video associated with past responses.
    result['response_video_mask'] = torch.zeros_like(result['video_mask'])
    spans = find_assistant_spans(ids[0].tolist(), processor.tokenizer)
    if len(spans) != n:
        raise ValueError('Assistant target alignment failed')
    labels = torch.full_like(ids, -100)
    for begin, end in spans:
        labels[0, begin:end] = ids[0, begin:end]
    apply_loss_mask(ids[0].tolist(), labels, spans, processor.tokenizer,
                    timing_only=False, stm=True)
    result['labels'] = labels
    result['video_chunk_size'] = 2
    result['position_ids'] = compute_position_ids(result, processor, model_type)
    return result


class StreamingDataset(Dataset):
    def __init__(self, manifest, video_root, processor, model_type, max_length=98304,
                 min_pixels=100352, max_pixels=150528):
        self.rows = []
        with open(manifest, encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    validate_record(row)
                    self.rows.append(row)
        if not self.rows:
            raise ValueError('Empty training manifest')
        self.video_root, self.processor, self.model_type = video_root, processor, model_type
        self.max_length, self.min_pixels, self.max_pixels = max_length, min_pixels, max_pixels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return encode_record(self.rows[index], self.processor, self.model_type,
                             self.video_root, self.max_length,
                             self.min_pixels, self.max_pixels)
