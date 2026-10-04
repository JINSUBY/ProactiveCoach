import os
import re
_TASK_MARKER = os.environ.get('THINKSTREAM_D3_TASK_MARKER', 'Phase')
TOP_MARKER_ALIASES = tuple(dict.fromkeys((_TASK_MARKER, 'Phase', 'Task')))
STYLES = {'atomic': ('<silent>', '<response>'), 'word': (' silent', ' respond'), 'named3': (' silent', ' response')}
_SPECS = {'atomic': {'markers': ('T', 'S', 'A'), 'sep': '', 'resp_sep': ' ', 'states': {'silent': '<silent>', 'response': '<response>'}}, 'word': {'markers': ('T', 'S', 'A'), 'sep': '', 'resp_sep': ' ', 'states': {'silent': ' silent', 'response': ' respond'}}, 'named3': {'markers': (_TASK_MARKER, 'Step', 'Action'), 'sep': ':', 'resp_sep': ' - ', 'states': {'silent': ' silent', 'standby': ' standby', 'response': ' response'}}}
STANDBY = 'standby'
STATES = ('silent', 'standby', 'response')
STYLE = os.environ.get('THINKSTREAM_D3_STYLE', 'atomic').lower()
if STYLE not in STYLES:
    raise ValueError(f'THINKSTREAM_D3_STYLE must be one of {sorted(STYLES)}, got {STYLE!r}')
SILENT, RESPONSE = STYLES[STYLE]
LEVELS = ('task', 'step', 'action')
MARKERS = _SPECS[STYLE]['markers']
MARKER_OF = dict(zip(LEVELS, MARKERS))
LEVEL_OF = dict(zip(MARKERS, LEVELS))
_EOS = ('<|im_end|>', '<|endoftext|>')

def _spec(style=None):
    style = (style or STYLE).lower()
    if style not in _SPECS:
        raise ValueError(f'unknown D3 style {style!r}; expected one of {sorted(STYLES)}')
    return _SPECS[style]

def _parse_line(marker, line, sp):
    for st, tok in sp['states'].items():
        head = f'{marker}{sp['sep']}{tok}'
        if st == 'response':
            if line.startswith(head):
                rest = line[len(head):]
                if rest.startswith(sp['resp_sep']) and rest[len(sp['resp_sep']):].strip():
                    return ('response', rest[len(sp['resp_sep']):].strip(), True)
                return ('silent', None, False)
        elif line == head:
            return (st, None, True)
    return ('silent', None, False)

def parse_d3_states(text, style=None):
    sp = _spec(style)
    mk = sp['markers']
    level_of = dict(zip(mk, LEVELS))
    accept = [(a, mk[0]) for a in TOP_MARKER_ALIASES] + [(mk[1], mk[1]), (mk[2], mk[2])]
    slots = {lv: {'state': 'silent', 'line': None} for lv in LEVELS}
    if not isinstance(text, str):
        return (slots, False)
    t = text
    for e in _EOS:
        t = t.replace(e, '')
    lines = [l.strip() for l in t.strip().split('\n')]
    lines = [l for l in lines if l]
    ok = len(lines) == 3
    seen = {}
    for l in lines:
        hit = next(((a, c) for a, c in accept if l.startswith(a + sp['sep'])), None)
        if hit is not None and hit[1] not in seen:
            seen[hit[1]] = (hit[0], l)
        else:
            ok = False
    for i, m in enumerate(mk):
        if m not in seen:
            ok = False
            continue
        alias, line = seen[m]
        if ok and lines[i] != line:
            ok = False
        st, val, lok = _parse_line(alias, line, sp)
        slots[level_of[m]] = {'state': st, 'line': val}
        ok = ok and lok
    return (slots, ok)
_LEVEL_WORD = re.compile('\\b(step|steps|action|actions|task|tasks|phase|phases|guide|tell|telling)\\b', re.I)
_LEVEL_ONLY = re.compile('\\b(step|steps|action|actions|task|tasks|phase|phases)\\b', re.I)
