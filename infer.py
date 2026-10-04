"""Run selected benchmark anchors with GT history, or a dense streaming demo."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--model-type', choices=['qwen35','qwen3vl'], default='qwen35')
    p.add_argument('--data', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--anchors', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--history', choices=['gt','generated'], default='gt')
    p.add_argument('--window-chunks', type=int, default=5)
    a = p.parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, Qwen3VLForConditionalGeneration
    from proactivecoach.inference import WindowPredictor
    from proactivecoach.data import validate_record
    from evaluation.parser import extract
    from evaluation.io import read_jsonl, index_unique
    rows = index_unique(read_jsonl(a.data), 'video_id')
    anchors = read_jsonl(a.anchors)
    index_unique(anchors, 'sample_id')
    for q in anchors:
        row = rows[q['video_id']]
        n = validate_record(row)
        if type(q['chunk']) is not int or not 0 <= q['chunk'] < n:
            raise ValueError('Invalid anchor chunk')
        if abs(q['time']-(row['video_start']+2*(q['chunk']+1))) > 1e-6:
            raise ValueError('Anchor time must be the original chunk endpoint')
    if a.history == 'generated':
        for vid in {q['video_id'] for q in anchors}:
            chunks = sorted(q['chunk'] for q in anchors if q['video_id']==vid)
            if chunks != list(range(validate_record(rows[vid]))):
                raise ValueError('Generated-history demos require every chunk, starting at zero')
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    processor = AutoProcessor.from_pretrained(a.model)
    Model = Qwen3_5ForConditionalGeneration if a.model_type=='qwen35' else Qwen3VLForConditionalGeneration
    model = Model.from_pretrained(a.model, dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda').eval()
    predictor = WindowPredictor(model, processor, a.model_type, a.window_chunks)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output,'x',encoding='utf-8') as f:
        for vid in dict.fromkeys(q['video_id'] for q in anchors):
            row, generated = rows[vid], []
            predictor.prepare(row, a.video_root)
            group = sorted((q for q in anchors if q['video_id']==vid), key=lambda q:q['chunk'])
            for q in group:
                history = [t['assistant'] for t in row['thoughts'][:q['chunk']]] if a.history=='gt' else generated
                raw, times = predictor.predict(q['chunk'], history)
                levels, route = extract(raw, 'joint', 'sparse')
                result = dict(sample_id=q['sample_id'],video_id=vid,chunk=q['chunk'],time=q['time'],
                    raw=raw, levels=levels, parse_route=route, frame_times=times, history=a.history)
                f.write(json.dumps(result)+'\n'); f.flush()
                generated.append(raw.split('<|im_end|>',1)[0])
                print(q['sample_id'], raw, flush=True)


if __name__ == '__main__':
    main()
