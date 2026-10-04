import json


def read_jsonl(path, allow_empty=False):
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows and not allow_empty:
        raise ValueError(f'Empty JSONL: {path}')
    return rows


def index_unique(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f'Duplicate {key}: {row[key]}')
        result[row[key]] = row
    return result


def aligned_samples(data, anchors, predictions):
    from proactivecoach.supervision import parse, LEVELS
    from .parser import extract
    rows = index_unique(data, 'video_id')
    index_unique(anchors, 'sample_id')
    pred = index_unique(predictions, 'sample_id')
    if set(pred) != {a['sample_id'] for a in anchors}:
        raise ValueError('Prediction IDs must exactly cover the supplied anchor manifest')
    result, seen = [], set()
    for a in anchors:
        row, p = rows[a['video_id']], pred[a['sample_id']]
        if any(p[k] != a[k] for k in ('video_id','chunk','time')):
            raise ValueError('Prediction coordinates disagree with anchor manifest')
        if p.get('history') != 'gt':
            raise ValueError('Benchmark evaluation requires GT-history predictions')
        k = a['chunk']
        if type(k) is not int or not 0 <= k < len(row['thoughts']):
            raise ValueError('Invalid anchor chunk')
        if abs(a['time']-(row['video_start']+2*(k+1))) > 1e-6:
            raise ValueError('Shifted anchor time')
        gt = parse(row['thoughts'][k]['assistant'])
        parsed, _ = extract(p['raw'], 'joint', 'sparse')
        levels = a['levels']
        if not levels or len(set(levels)) != len(levels) or set(levels)-set(LEVELS):
            raise ValueError('Invalid selected levels')
        for lv in levels:
            key = (a['video_id'], k, lv)
            if key in seen:
                raise ValueError('Duplicate video/chunk/level anchor')
            seen.add(key)
            ref = ' / '.join(e[2] for e in gt if e[0]==lv)
            result.append(dict(sample_id=a['sample_id'], video_id=a['video_id'], level=lv,
                time=a['time'], reference=ref, prediction=parsed[lv]['text'],
                decision=parsed[lv]['decision'], context=row['conversations'][0]['content']))
    return result
