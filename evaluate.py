"""Score a complete anchor prediction file; optionally export PQS judge requests."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--anchors',required=True)
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--judgments')
    p.add_argument('--export-judge-requests')
    p.add_argument('--skip-semantic',action='store_true')
    p.add_argument('--encoder',default='sentence-transformers/all-mpnet-base-v2')
    a = p.parse_args()
    from evaluation.io import aligned_samples,read_jsonl,index_unique
    from evaluation.metrics import evaluate,judge_request
    samples = aligned_samples(read_jsonl(a.data),read_jsonl(a.anchors),read_jsonl(a.predictions))
    if a.export_judge_requests:
        requests = {}
        for s in samples:
            if s['reference'] and s['decision']=='response':
                r = judge_request(s); requests[r['id']] = r
        with open(a.export_judge_requests,'x',encoding='utf-8') as f:
            for r in requests.values():
                f.write(json.dumps(r)+'\n')
    judgments = index_unique(read_jsonl(a.judgments,allow_empty=True),'id') if a.judgments else {}
    encoder = None
    if not a.skip_semantic:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(a.encoder,device='cpu')
    result = evaluate(samples,encoder,judgments)
    with open(a.output,'x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
