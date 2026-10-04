import argparse
import csv
import json
from collections import defaultdict

LEVELS = ('phase','step','action')
METRICS = ('alignment','coverage','relevance','quality')


def aggregate(rows):
    values, counts, seen = defaultdict(list), defaultdict(lambda:dict(rated=0,unratable=0,missing=0)), set()
    for row in rows:
        if not row['case_id'] or not row['reviewer_id']:
            raise ValueError('case_id and reviewer_id are required')
        lv, metric = row['level'], row['metric']
        if lv not in LEVELS or metric not in METRICS:
            raise ValueError('Unknown level or metric')
        key = (row['case_id'],row['reviewer_id'],lv,metric)
        if key in seen:
            raise ValueError('Duplicate rating')
        seen.add(key)
        group, value = lv+'/'+metric, row['score'].strip()
        if value in ('','U'):
            counts[group]['unratable' if value=='U' else 'missing'] += 1
        else:
            if value not in ('0','1','2','3','4','5'):
                raise ValueError('Scores must be 0..5, U, or blank')
            values[group].append(int(value)); counts[group]['rated'] += 1
    cases = {(r[0],r[1]) for r in seen}
    if len(seen) != len(cases)*len(LEVELS)*len(METRICS):
        raise ValueError('Each case/reviewer requires all 12 rows, including missing ratings')
    return {group:dict(**c,mean=sum(values[group])/len(values[group]) if values[group] else None)
            for group,c in sorted(counts.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ratings')
    p.add_argument('--output',required=True)
    a = p.parse_args()
    with open(a.ratings,encoding='utf-8',newline='') as f:
        result = aggregate(list(csv.DictReader(f)))
    with open(a.output,'x',encoding='utf-8') as f:
        json.dump(result,f,indent=2); f.write('\n')


if __name__=='__main__':
    main()
