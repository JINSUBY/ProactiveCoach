# Adapted from the training implementation. See LICENSE.
import os
import random
import re
LEVELS = ('phase', 'step', 'action')
ENTRY = re.compile('(phase|step|action) (\\d+(?:-\\d+){0,2}): (.*?)(?= / (?:phase|step|action) \\d|$)')
PROMPT = 'You are a proactive procedural assistant watching a first-person video stream, one new frame every second. After each one-second chunk, decide whether to guide. Output exactly <silent> when no new guidance is needed. Otherwise output <response> followed only by the guidance entries needed now, separated by " / ". Each entry is "phase P: guidance", "step P-S: guidance", or "action P-S-A: guidance". Positive numbered IDs express parent-child relationships; number units in their temporal order within this clip and restart child numbers under each parent. Phase is a sub-goal spanning steps; Step is a coherent operation; Action is a small manipulation. Put entries in Phase, Step, Action order. Announce the next unit of a level at the first one-second decision after the previous announced unit of that level has finished. For the first unit, guide at the first available decision. Do not repeat a guide or guide an already finished unit. Omit inactive levels rather than printing their silent states. Use the goal, observed frames and your prior outputs; no future frames are available. There is no think block or other wrapper. Example: <response> phase 1: Prepare the tomato. / step 1-1: Wash and dry it. / action 1-1-1: Wash the tomato.'
PLAN_PROMPT = '\nAn untimed guide inventory is provided with the user goal. Its IDs and wording are supplied, but its timing and completion states are not. Select the entries to announce from visual progress. Copy the supplied wording verbatim. The inventory stays fixed throughout the clip; its presence is not a cue to speak.'
if os.environ.get('THINKSTREAM_CHUNK_SIZE', '1') in ('2', '2.0'):
    PROMPT = PROMPT.replace('After each one-second chunk', 'After each two-second chunk containing two frames')
    PROMPT = PROMPT.replace('first one-second decision', 'first two-second decision')
if os.environ.get('THINKSTREAM_SPARSE_PRESERVE_LEGACY') == '1':
    PROMPT = PROMPT.replace('Announce the next unit of a level at the first two-second decision after the previous announced unit of that level has finished. For the first unit, guide at the first available decision. ', 'Use the observed procedural progress to decide when the next guidance is needed. ')

def parse(text):
    if text == '<silent>':
        return []
    if not text.startswith('<response> '):
        raise ValueError('Malformed sparse response: ' + text[:120])
    body = text[len('<response> '):]
    matches = list(ENTRY.finditer(body))
    if not matches or ' / '.join((m.group(0) for m in matches)) != body:
        raise ValueError('Malformed guidance entries: ' + body[:120])
    entries = []
    for m in matches:
        level, uid, guide = m.groups()
        if len(uid.split('-')) != LEVELS.index(level) + 1 or not guide.strip():
            raise ValueError('Invalid hierarchy identifier or empty guide')
        if any((int(n) <= 0 for n in uid.split('-'))):
            raise ValueError('Identifiers must be positive')
        entries.append((level, uid, guide))
    if [LEVELS.index(e[0]) for e in entries] != sorted((LEVELS.index(e[0]) for e in entries)):
        raise ValueError('Levels out of order')
    if len({e[:2] for e in entries}) != len(entries):
        raise ValueError('Duplicate unit in one response')
    return entries

def apply_loss_mask(ids, labels, spans, tokenizer, *, timing_only=False, stm=True):
    records = []
    for start, end in spans:
        seq = ids[start:end]
        raw = tokenizer.decode(seq, skip_special_tokens=False)
        body = raw.split('<|im_end|>', 1)[0]
        entries = parse(body)
        encoded = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
        if encoded['input_ids'] != seq:
            raise ValueError('Assistant token roundtrip changed; refusing incorrect loss masking')
        records.append((start, body, bool(entries), encoded['offset_mapping']))
    active = [i for i, r in enumerate(records) if i == 0 or r[2] or r[2] != records[i - 1][2]]
    continuing = [i for i in range(len(records)) if i not in set(active)]
    keep = set(active)
    rng = random.Random(len(ids) * 1000003 + (spans[0][0] if spans else 0))
    keep.update(rng.sample(continuing, min(len(continuing), max(1, len(active)))))
    masked_decisions = masked_wording = 0
    for i, (start, body, response, offsets) in enumerate(records):
        ranges = []
        if stm and i not in keep:
            ranges.append((1, len('<silent>') - 1))
            masked_decisions += 1
        if timing_only and response:
            prefix = len('<response> ')
            ranges += [(prefix + m.start(3), prefix + m.end(3)) for m in ENTRY.finditer(body[prefix:])]
        for j, (a, b) in enumerate(offsets):
            if any((a < hi and lo < b for lo, hi in ranges)):
                labels[0, start + j] = -100
                masked_wording += timing_only and response
    return dict(chunks=len(records), responses=sum((r[2] for r in records)), masked_silent=masked_decisions, masked_wording=masked_wording, timing_only=timing_only, stm=stm)
