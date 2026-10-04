# Adapted from the training implementation. See LICENSE.
import os
import torch
import transformers
from dataclasses import dataclass
from typing import Dict, Sequence
IGNORE_INDEX = -100

def pad_and_cat(tensor_list):
    max_length = max((tensor.shape[2] for tensor in tensor_list))
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), 'constant', 1)
        padded_tensors.append(padded_tensor)
    stacked_tensor = torch.cat(padded_tensors, dim=1)
    return stacked_tensor

def _cat_visual_tensors(tensors):
    if os.environ.get('THINKSTREAM_VISUAL_IPC_BF16') == '1':
        tensors = [tensor.to(torch.bfloat16) for tensor in tensors]
    return torch.cat(tensors, dim=0)

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    vocab_size: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, video_masks = tuple(([instance[key] for instance in instances] for key in ('input_ids', 'labels', 'position_ids', 'video_mask')))
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        video_masks = [ids.squeeze(0) for ids in video_masks]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        video_masks = torch.nn.utils.rnn.pad_sequence(video_masks, batch_first=True, padding_value=0)
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, :self.tokenizer.model_max_length]
        video_masks = video_masks[:, :self.tokenizer.model_max_length]
        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), video_mask=video_masks)
        if all(('response_video_mask' in inst for inst in instances)):
            rvm = torch.nn.utils.rnn.pad_sequence([inst['response_video_mask'].squeeze(0) for inst in instances], batch_first=True, padding_value=0)
            batch['response_video_mask'] = rvm[:, :self.tokenizer.model_max_length]
        if all(('token_loss_weight' in inst for inst in instances)):
            tlw = torch.nn.utils.rnn.pad_sequence([inst['token_loss_weight'].squeeze(0) for inst in instances], batch_first=True, padding_value=1.0)
            batch['token_loss_weight'] = tlw[:, :self.tokenizer.model_max_length]
        if all(('hidden_watch_mask' in inst for inst in instances)):
            hwm = torch.nn.utils.rnn.pad_sequence([inst['hidden_watch_mask'].squeeze(0) for inst in instances], batch_first=True, padding_value=False)
            batch['hidden_watch_mask'] = hwm[:, :self.tokenizer.model_max_length]
        images = list((instance['pixel_values'] for instance in instances if 'pixel_values' in instance))
        videos = list((instance['pixel_values_videos'] for instance in instances if 'pixel_values_videos' in instance))
        if len(images) != 0:
            concat_images = _cat_visual_tensors(images)
            grid_thw = [instance['image_grid_thw'] for instance in instances if 'image_grid_thw' in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None
        if len(videos) != 0:
            concat_videos = _cat_visual_tensors(videos)
            video_grid_thw = [instance['video_grid_thw'] for instance in instances if 'video_grid_thw' in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None
        batch['pixel_values'] = concat_images
        batch['image_grid_thw'] = grid_thw
        batch['pixel_values_videos'] = concat_videos
        batch['video_grid_thw'] = video_grid_thw
        batch['position_ids'] = position_ids
        _bucket = int(os.environ.get('THINKSTREAM_PAD_BUCKET', '0'))
        if _bucket > 0:
            L_cur = batch['input_ids'].shape[1]
            L_new = min((L_cur + _bucket - 1) // _bucket * _bucket, self.tokenizer.model_max_length)
            pad_n = L_new - L_cur
            if pad_n > 0:
                F_pad = torch.nn.functional.pad
                batch['input_ids'] = F_pad(batch['input_ids'], (0, pad_n), value=self.tokenizer.pad_token_id)
                batch['labels'] = F_pad(batch['labels'], (0, pad_n), value=IGNORE_INDEX)
                batch['attention_mask'] = F_pad(batch['attention_mask'], (0, pad_n), value=False)
                batch['video_mask'] = F_pad(batch['video_mask'], (0, pad_n), value=0)
                batch['position_ids'] = F_pad(batch['position_ids'], (0, pad_n), value=1)
                for k, v in (('response_video_mask', 0), ('hidden_watch_mask', False), ('token_loss_weight', 1.0)):
                    if k in batch:
                        batch[k] = F_pad(batch[k], (0, pad_n), value=v)
        if os.environ.get('THINKSTREAM_CE_WEIGHT_OFF', '') == '1':
            batch['ce_weight'] = torch.ones(self.vocab_size)
            return batch
        _term_ids = self.tokenizer.convert_tokens_to_ids(['<silent>', '<response>', '<ongoing>', '<done>'])
        if any((t is None or t == self.tokenizer.unk_token_id for t in _term_ids)):
            _words = (' silent', ' respond', ' ongoing', ' done', 'silent', 'response', 'ongo', 'done')
            _term_ids = sorted({self.tokenizer.encode(w, add_special_tokens=False)[0] for w in _words})
        silent_id, response_id = (_term_ids[0], _term_ids[1])
        _counts = [(tid, (batch['labels'] == tid).sum()) for tid in _term_ids]
        _present = [(tid, n) for tid, n in _counts if int(n) > 0]
        ce_weight = torch.ones(self.vocab_size)
        eps = 0.001
        if _present:
            total_n = sum((n for _, n in _present))
            k = len(_present)
            for tid, n in _present:
                ce_weight[tid] = total_n / (k * n + eps)
        batch['ce_weight'] = torch.clamp(ce_weight, 0, 20)
        _rvm = batch.get('response_video_mask')
        _vt = int(video_masks.sum())
        _rvm_cov = float(_rvm.sum()) / _vt if _rvm is not None and _vt else -1.0
        print(input_ids.shape, batch['ce_weight'][_term_ids], f'rvm_cov={_rvm_cov:.3f}')
        return batch
