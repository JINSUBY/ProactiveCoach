# Adapted from the training implementation. See LICENSE.
import os, logging
import numpy as np
import torch
from typing import Optional, List, Tuple
from qwen_vl_utils import process_vision_info

class _AvVideoMeta:

    def __init__(self, fps, n, dur):
        self.average_fps = fps
        self.num_frames = n
        self.duration_seconds = dur

class VideoDecoder:

    def __init__(self, path):
        import av
        with av.open(path) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate) if s.average_rate else 30.0
            if c.duration:
                dur = float(c.duration) / av.time_base
            elif s.duration and s.time_base:
                dur = float(s.duration * s.time_base)
            else:
                dur = 0.0
            n = s.frames or int(round(fps * dur))
        self.metadata = _AvVideoMeta(fps, n, dur)

def _decode_packets_av(container, stream, path):
    import av
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError as exc:
            logging.warning('Unreadable video packet: %s pts=%s; %s', path, packet.pts, exc)
            continue
        yield from frames

def _read_video_av(ele):
    import av
    import numpy as np
    from qwen_vl_utils import vision_process as _vp
    path = ele['video']
    if path.startswith('file://'):
        path = path[7:]
    v_start = float(ele.get('video_start') or 0.0)
    v_end = ele.get('video_end', None)
    with av.open(path) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 2
        fps = float(s.average_rate) if s.average_rate else 30.0
        if v_end is None:
            if c.duration:
                v_end = float(c.duration) / av.time_base
            else:
                v_end = float(s.duration * s.time_base) if s.duration else 0.0
        total_frames = max(int(round((v_end - v_start) * fps)), 1)
        nframes = _vp.smart_nframes(ele, total_frames=total_frames, video_fps=fps)
        targets = np.linspace(v_start, v_end, nframes, endpoint=False)
        right_aligned = os.environ.get('THINKSTREAM_PROMPT') == 'SPARSE_GUIDE' and os.environ.get('THINKSTREAM_SPARSE_PRESERVE_LEGACY') != '1'
        if right_aligned:
            targets += (v_end - v_start) / nframes
        seek_pts = 0 if ele.get('_av_from_start') else int(v_start / (s.time_base or 1 / fps))
        try:
            c.seek(seek_pts, stream=s, backward=True)
        except Exception:
            pass
        frames = [None] * nframes
        frame_indices = [None] * nframes
        ti = 0
        previous_frame = previous_time = None
        decoded = _decode_packets_av(c, s, path) if os.environ.get('THINKSTREAM_PROMPT') == 'SPARSE_GUIDE' else c.decode(s)
        for fr in decoded:
            if ti >= nframes:
                break
            t = float(fr.pts * s.time_base) if fr.pts is not None else None
            if t is None:
                continue
            if right_aligned:
                while ti < nframes and t > targets[ti] + 1e-07:
                    if previous_frame is None:
                        if not ele.get('_av_from_start'):
                            logging.warning('Retry causal decode from origin: %s at %s', path, v_start)
                            return _read_video_av(dict(ele, _av_from_start=True))
                        raise RuntimeError(f'No causal frame before decision {targets[ti]}: {path}; first PTS={t}')
                    frames[ti] = previous_frame.to_ndarray(format='rgb24')
                    frame_indices[ti] = round(previous_time * fps)
                    ti += 1
                previous_frame, previous_time = (fr, t)
                continue
            while ti < nframes and t >= targets[ti] - 0.5 / fps:
                frames[ti] = fr.to_ndarray(format='rgb24')
                frame_indices[ti] = round(t * fps)
                ti += 1
        if right_aligned and previous_frame is not None:
            while ti < nframes:
                assert previous_time <= targets[ti] + 1e-07
                frames[ti] = previous_frame.to_ndarray(format='rgb24')
                frame_indices[ti] = round(previous_time * fps)
                ti += 1
        last = next((f for f in frames if f is not None), None)
        if last is None:
            if right_aligned and (not ele.get('_av_from_start')):
                return _read_video_av(dict(ele, _av_from_start=True))
            raise RuntimeError(f'av reader: no frames decoded from {path}')
        for i in range(nframes):
            if frames[i] is None:
                frames[i] = last
                frame_indices[i] = frame_indices[i - 1]
            last = frames[i]
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    idx = torch.arange(nframes)
    if right_aligned:
        idx = torch.tensor(frame_indices)
    sample_fps = nframes / max(total_frames, 1e-06) * fps
    meta = dict(fps=fps, frames_indices=idx, total_num_frames=total_frames, video_backend='av_stream')
    return (video, meta, sample_fps)
try:
    from qwen_vl_utils import vision_process as _vp_mod
    _vp_mod.VIDEO_READER_BACKENDS['torchvision'] = _read_video_av
except Exception:
    pass

def _get_video_pixels(processor):
    vp = processor.video_processor
    min_px = getattr(vp, 'min_pixels', None)
    max_px = getattr(vp, 'max_pixels', None)
    if min_px is None or max_px is None:
        size = getattr(vp, 'size', {})
        min_px = min_px or size.get('shortest_edge')
        max_px = max_px or size.get('longest_edge')
    if min_px is None or max_px is None:
        raise ValueError(f'Cannot resolve video pixel limits from processor.video_processor: min_pixels={min_px}, max_pixels={max_px}, size={getattr(vp, 'size', 'N/A')}')
    return (min_px, max_px)

def _resolve_vit_patch_size(processor) -> int:
    return getattr(processor.video_processor, 'patch_size', getattr(processor.image_processor, 'patch_size', 14))

def load_video_frames(video_path: str, video_start: float, video_end: float, total_nframes: int, frames_per_chunk: int, num_chunks: int, *, min_pixels: Optional[int]=None, max_pixels: Optional[int]=None, processor=None, vit_patch_size: Optional[int]=None, model_type: str) -> Tuple[List[torch.Tensor], dict, Optional[List[dict]]]:
    is_qwen3vl = model_type in ('qwen3vl', 'qwen35')
    if min_pixels is None or max_pixels is None:
        if processor is None:
            raise ValueError('Either (min_pixels, max_pixels) or processor must be provided.')
        _min, _max = _get_video_pixels(processor)
        min_pixels = min_pixels or _min
        max_pixels = max_pixels or _max
    if video_end <= video_start:
        video_end = video_start + 0.001
    ghost_message = [{'role': 'user', 'content': [{'type': 'video', 'video': video_path, 'video_start': video_start, 'video_end': video_end, 'nframes': total_nframes, 'min_pixels': min_pixels, 'max_pixels': max_pixels}]}]
    pvi_kwargs: dict = dict(return_video_kwargs=True)
    if vit_patch_size is not None:
        pvi_kwargs['image_patch_size'] = vit_patch_size
    elif processor is not None:
        pvi_kwargs['image_patch_size'] = _resolve_vit_patch_size(processor)
    if is_qwen3vl:
        pvi_kwargs['return_video_metadata'] = True
    vision_reader = process_vision_info
    _, video_inputs_list, video_kwargs = vision_reader(ghost_message, **pvi_kwargs)
    if is_qwen3vl:
        big_video_tensor, video_metadata = video_inputs_list[0]
    else:
        big_video_tensor = video_inputs_list[0]
        video_metadata = None
    split_videos = list(torch.split(big_video_tensor, frames_per_chunk, dim=0))
    if len(split_videos) > num_chunks:
        split_videos = split_videos[:num_chunks]
    chunk_metadatas: Optional[List[dict]] = None
    if is_qwen3vl and video_metadata is not None:
        all_indices = video_metadata['frames_indices']
        if isinstance(all_indices, torch.Tensor):
            chunk_idx_splits = list(torch.split(all_indices, frames_per_chunk))
        else:
            chunk_idx_splits = [all_indices[i:i + frames_per_chunk] for i in range(0, len(all_indices), frames_per_chunk)]
        chunk_idx_splits = chunk_idx_splits[:len(split_videos)]
        chunk_metadatas = [{**video_metadata, 'frames_indices': ci} for ci in chunk_idx_splits]
    elif 'fps' in video_kwargs and isinstance(video_kwargs['fps'], list):
        video_kwargs['fps'] = [video_kwargs['fps'][0]] * len(split_videos)
    return (split_videos, video_kwargs, chunk_metadatas)
