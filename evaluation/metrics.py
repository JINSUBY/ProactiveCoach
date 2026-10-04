"""Anchor decision scores, timestamp-aware semantic matching, and PQS."""
import collections
import hashlib
import math

from .judge_prompt import JUDGE_PROMPT, DIMS


def judge_request(sample):
    prompt = JUDGE_PROMPT.format(golden=sample['reference'], model=sample['prediction'],
                               task_context=sample['context'])
    return dict(id=hashlib.sha256(('gpt-5.2\0'+prompt).encode()).hexdigest(),
                judge='gpt-5.2', prompt=prompt)


def decision_scores(samples):
    c = collections.Counter(N=len(samples))
    for s in samples:
        response = bool(s['reference'])
        if s['decision']=='invalid':
            c['invalid_response' if response else 'invalid_silent'] += 1
        else:
            predicted = s['decision']=='response'
            c['TP' if response and predicted else 'FN' if response else 'FP' if predicted else 'TN'] += 1
    tp, fp, tn, fn, ir, iz = [c[k] for k in ('TP','FP','TN','FN','invalid_response','invalid_silent')]
    rf1 = 2*tp/max(1,2*tp+fp+fn+ir)
    sf1 = 2*tn/max(1,2*tn+fp+fn+iz)
    return dict(counts=dict(c), RF1=100*rf1, SF1=100*sf1,
                gF1=100*math.sqrt(rf1*sf1), invalid=ir+iz)


def semantic_score(samples, encoder):
    """One video and one level. Silent anchors remain matching candidates."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    rows = sorted(samples, key=lambda s:s['time'])
    times = np.asarray([s['time'] for s in rows], dtype=np.float64)
    ref_times = [s['time'] for s in rows if s['reference']]
    boundaries = [(a+b)/2 for a,b in zip(ref_times,ref_times[1:])]
    segments = np.searchsorted(boundaries,times,side='right')
    credit = 0.0
    predicted = sum(bool(s['prediction']) or s['decision']=='invalid' for s in rows)
    references = sum(bool(s['reference']) for s in rows)
    for sid in sorted(set(segments)):
        indices = np.flatnonzero(segments==sid)
        gens = [i for i in indices if rows[i]['prediction'] or rows[i]['decision']=='invalid']
        refs = [j for j in indices if rows[j]['reference']]
        if not gens:
            continue
        sim = np.zeros((len(gens),len(indices)),dtype=np.float32)
        valid = [i for i in gens if rows[i]['decision']!='invalid']
        if valid and refs:
            texts = [rows[i]['prediction'] for i in valid]+[rows[j]['reference'] for j in refs]
            emb = np.asarray(encoder.encode(texts),dtype=np.float32)
            emb /= np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
            scores = np.clip(np.abs(emb[:len(valid)]@emb[len(valid):].T),0,1)
            for ii,i in enumerate(valid):
                for jj,j in enumerate(refs):
                    sim[gens.index(i),int(np.flatnonzero(indices==j)[0])] = scores[ii,jj]
        distance = (1-np.exp(-.01*(times[gens,None]-times[None,indices])**2)).astype(np.float32)
        cost = 1-sim+distance
        # The sparse assignment routine omits numerical zero edges.
        cost[cost==0] = 1e-10
        rr,cc = min_weight_full_bipartite_matching(csr_matrix(cost))
        credit += float(sim[rr,cc].sum())
    sp = credit/predicted if predicted else 0
    sr = credit/references if references else 0
    return dict(sP=100*sp,sR=100*sr,sF1=200*credit/max(1,predicted+references))


def pqs_score(samples, judgments):
    total, required, missing = 0.0, set(), set()
    for s in samples:
        if not s['reference'] and s['decision']=='silent':
            total += 1
        elif s['reference'] and s['decision']=='response':
            req = judge_request(s)
            key = req['id']; required.add(key)
            if key not in judgments:
                missing.add(key)
                continue
            j = judgments[key]
            scores = j['scores']
            if j['judge'] != req['judge'] or any(
                type(scores.get(d)) not in (int,float) or not 1 <= scores[d] <= 5 for d in DIMS):
                raise ValueError('Invalid PQS judgment or judge identity')
            total += (sum(scores[d] for d in DIMS)/4-1)/4
    return dict(PQS=None if missing else 100*total/max(1,len(samples)),
                judge_pairs=len(required), missing_judgments=len(missing))


def evaluate(samples, encoder=None, judgments=None):
    result = {}
    for lv in ('phase','step','action'):
        group = [s for s in samples if s['level']==lv]
        if not group:
            continue
        score = decision_scores(group)
        score.update(pqs_score(group,judgments or {}))
        score.update(sP=None,sR=None,sF1=None)
        if encoder is not None:
            videos = collections.defaultdict(list)
            for s in group:
                videos[s['video_id']].append(s)
            per_video = [semantic_score(v,encoder) for v in videos.values()]
            score.update({k:sum(v[k] for v in per_video)/len(per_video) for k in ('sP','sR','sF1')})
        score['Avg'] = (score['sF1']+score['gF1']+score['PQS'])/3 if (
            score['sF1'] is not None and score['PQS'] is not None) else None
        result[lv] = score
    return result
