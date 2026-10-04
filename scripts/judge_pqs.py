"""Run exported PQS requests. Requires an API key; API calls incur charges."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.io import read_jsonl,index_unique
from evaluation.judge_prompt import DIMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    requests = read_jsonl(a.requests,allow_empty=True)
    index_unique(requests,'id')
    done = index_unique(read_jsonl(a.output),'id') if Path(a.output).exists() and Path(a.output).stat().st_size else {}
    key = os.environ['OPENAI_API_KEY'] if any(r['id'] not in done for r in requests) else ''
    with open(a.output,'a',encoding='utf-8') as f:
        for r in requests:
            if r['id'] in done:
                continue
            if r['judge'] != 'gpt-5.2':
                raise ValueError('Expected the evaluation judge gpt-5.2')
            payload = dict(model=r['judge'],messages=[dict(role='user',content=r['prompt'])],
                           response_format={'type':'json_object'})
            req = urllib.request.Request('https://api.openai.com/v1/chat/completions',
                data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as response:
                text = json.load(response)['choices'][0]['message']['content']
            scores = json.loads(text)
            if any(type(scores.get(d)) not in (int,float) or not 1 <= scores[d] <= 5 for d in DIMS):
                raise ValueError('Judge returned an invalid score; no result stored')
            result = dict(id=r['id'],judge=r['judge'],scores={d:scores[d] for d in DIMS},
                          reasoning=scores.get('reasoning',''))
            f.write(json.dumps(result)+'\n'); f.flush()
            print(r['id'],flush=True)


if __name__ == '__main__':
    main()
