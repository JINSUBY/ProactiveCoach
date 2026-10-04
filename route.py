import argparse
import json


def main():
    p = argparse.ArgumentParser(description='Replay LLM level routing on saved guidance packets')
    p.add_argument('--model',required=True)
    p.add_argument('--predictions',required=True)
    p.add_argument('--requests',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--initial-level',choices=['Phase','Step','Action'],default='Step')
    p.add_argument('--device',default='cuda:0')
    a = p.parse_args()
    from evaluation.io import read_jsonl,index_unique
    from proactivecoach.router import AsyncRouter,select_guidance
    packets = read_jsonl(a.predictions)
    index_unique(packets,'sample_id')
    requests = read_jsonl(a.requests)
    if {r['video_id'] for r in requests}-{r['video_id'] for r in packets}:
        raise ValueError('A request refers to a missing video')
    router = AsyncRouter(a.model,device=a.device)
    try:
        with open(a.output,'x',encoding='utf-8') as f:
            for vid in dict.fromkeys(p['video_id'] for p in packets):
                level, cursor = a.initial_level, 0
                turns = sorted((r for r in requests if r['video_id']==vid),key=lambda r:r['time'])
                for packet in sorted((p for p in packets if p['video_id']==vid),key=lambda p:p['time']):
                    applied = []
                    while cursor < len(turns) and turns[cursor]['time'] <= packet['time']:
                        request = turns[cursor]
                        result = router.submit([(level,request['request'])]).result()['outputs'][0]
                        if result['level'] is not None:
                            level = result['level']
                        applied.append(dict(request=request,router=result));cursor += 1
                    result = dict(sample_id=packet['sample_id'],video_id=vid,time=packet['time'],
                        **select_guidance(packet['raw'],level),requests_applied=applied)
                    f.write(json.dumps(result)+'\n');f.flush()
    finally:
        router.close()


if __name__=='__main__':
    main()
