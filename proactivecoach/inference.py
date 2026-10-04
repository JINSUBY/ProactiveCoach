"""Chronological text history with a freshly replayed recent visual window."""
from pathlib import Path

from .data import validate_record
from .supervision import PROMPT
from .template import CHAT_TEMPLATE
from .processing import compute_position_ids
from .video import load_video_frames


def window_plan(row, chunk, window=5):
    n = validate_record(row)
    if not 0 <= chunk < n or window < 1:
        raise ValueError('Invalid chunk or window')
    first = max(0, chunk-window+1)
    return first, [row['video_start']+i for i in range(2*first, 2*(chunk+1))]


def user_messages(row, chunk, visual):
    start = row['video_start']+2*chunk
    content = [{'type': 'video', 'video': row['video_path']}] if visual else []
    queries = [q['content'] for q in row['conversations'] if start <= q['timestamp'] < start+2]
    if queries:
        content.append({'type': 'text', 'text': '\n'+'\n'.join(queries)})
    return ([{'role': 'system', 'content': PROMPT}] if chunk == 0 else []) + [
        {'role': 'user', 'content': content}]


def replay_anchor(prefix, recent, forward):
    if not recent or recent[-1][1] is not None:
        raise ValueError('Current target must not enter the prediction context')
    cache, position, out = None, 0, None
    if prefix is not None:
        out, position = forward(prefix, cache, position)
        cache = out.past_key_values
    for visual, past_output in recent:
        out, position = forward(visual, cache, position)
        cache = out.past_key_values
        if past_output is not None:
            out, position = forward(past_output, cache, position)
            cache = out.past_key_values
    return out, position


class WindowPredictor:
    def __init__(self, model, processor, model_type='qwen35', window=5, max_new_tokens=384):
        self.model, self.processor = model, processor
        self.model_type, self.window, self.max_new_tokens = model_type, window, max_new_tokens
        self.device = next(model.parameters()).device

    def tokens(self, text):
        return self.processor.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids']

    def prepare(self, row, video_root):
        n = validate_record(row)
        path = Path(row['video_path'])
        if not path.is_absolute():
            path = Path(video_root)/path
        self.row = row
        self.videos, self.kwargs, self.metadata = load_video_frames(
            str(path), row['video_start'], row['video_end'], 2*n, 2, n,
            min_pixels=100352, max_pixels=150528,
            processor=self.processor, model_type=self.model_type)
        self.visual = {}

    def visual_input(self, chunk):
        if chunk not in self.visual:
            text = self.processor.apply_chat_template(
                user_messages(self.row, chunk, True), tokenize=False,
                add_generation_prompt=True, chat_template=CHAT_TEMPLATE)
            kwargs = {k: [v[0]] if isinstance(v, list) else v for k,v in self.kwargs.items()}
            value = dict(self.processor(text=[text], videos=[self.videos[chunk]],
                video_metadata=[self.metadata[chunk]] if self.metadata else None,
                do_resize=False, return_tensors='pt', **kwargs))
            value['video_chunk_size'] = 2
            value['position_ids'] = compute_position_ids(value, self.processor, self.model_type)
            self.visual[chunk] = value
        return self.visual[chunk]

    def forward(self, inputs, cache, position):
        import torch
        tokens = inputs['input_ids'].to(self.device)
        pos = inputs.get('position_ids')
        if pos is None:
            pos = torch.arange(tokens.shape[1]).view(1,1,-1).expand(3,1,-1)
        pos = pos.to(self.device)+position
        kwargs = dict(input_ids=tokens, position_ids=pos, past_key_values=cache,
                      use_cache=True, logits_to_keep=1)
        for key in ('pixel_values_videos', 'video_grid_thw'):
            if key in inputs:
                kwargs[key] = inputs[key].to(self.device)
        return self.model(**kwargs), int(pos.max())+1

    def predict(self, chunk, past_outputs):
        import torch
        if len(past_outputs) != chunk:
            raise ValueError('Supply exactly the strictly-past chunk outputs')
        first, times = window_plan(self.row, chunk, self.window)
        old = []
        for j in range(first):
            text = self.processor.apply_chat_template(user_messages(self.row, j, False),
                tokenize=False, add_generation_prompt=True, chat_template=CHAT_TEMPLATE)
            old.extend([self.tokens(text), self.tokens(past_outputs[j]+'<|im_end|>')])
        prefix = {'input_ids': torch.cat(old, dim=1)} if old else None
        recent = [(self.visual_input(j), {'input_ids': self.tokens(past_outputs[j]+'<|im_end|>')}
                   if j < chunk else None) for j in range(first, chunk+1)]
        eos = {self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')}
        configured = self.model.generation_config.eos_token_id
        eos.update(configured if isinstance(configured, list) else [configured])
        with torch.inference_mode():
            out, position = replay_anchor(prefix, recent, self.forward)
            generated = []
            for _ in range(self.max_new_tokens):
                token = int(out.logits[0,-1].argmax())
                generated.append(token)
                if token in eos:
                    break
                out, position = self.forward({'input_ids': torch.tensor([[token]])},
                                             out.past_key_values, position)
        self.visual = {j:v for j,v in self.visual.items() if j >= first}
        return self.processor.tokenizer.decode(generated, skip_special_tokens=False), times
