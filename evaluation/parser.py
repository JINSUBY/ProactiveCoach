"""Extract guidance without equating a known alternate serialization with failure.

No GT text, timing or score is used. Ambiguous/contradictory responses remain invalid.
"""
import re
from .legacy_format import parse_d3_states
from proactivecoach.supervision import parse as parse_sparse

LEVELS = ('phase', 'step', 'action')


def extract(raw, mode, trained_format):
    body = raw.split('<|im_end|>', 1)[0].strip()
    invalid = {lv: dict(decision='invalid', text='') for lv in LEVELS}
    def slots(texts):
        return {lv: dict(decision='response' if texts.get(lv) else 'silent', text=texts.get(lv, '')) for lv in LEVELS}
    def named(text):
        parsed, valid = parse_d3_states(text, style='named3')
        if valid:
            return slots({lv: parsed[k]['line'] if parsed[k]['state']=='response' else ''
                          for lv,k in [('phase','task'),('step','step'),('action','action')]})
    def sparse(text):
        try:
            entries = parse_sparse(text)
        except ValueError:
            return None
        return slots({lv: ' / '.join(e[2] for e in entries if e[0]==lv) for lv in LEVELS})
    first = named if trained_format=='named3' else sparse
    result = first(body)
    if result is not None:
        return result, 'trained-format'
    # A whole-output global silence marker is unambiguous for every requested level.
    normalized = re.sub(r'^<\s*(silent|response)\s*>', lambda m:'<'+m[1]+'>', body)
    if normalized == '<silent>':
        return slots({}), 'explicit-global-silence'
    for name, parser in [('sparse', sparse), ('named3', named)]:
        result = parser(normalized)
        if result is not None:
            return result, 'alternate-'+name
    # A selected-Step call may state its response without the two unused level slots.
    # Do not infer Step content from a Phase/Action response or contradictory silence.
    if mode == 'step':
        m = re.fullmatch(r'<response>\s+(?:response\s*[-:]\s*)?(.+)', normalized, flags=re.S)
        if m:
            guide = m[1].strip()
            if (guide and not re.search(r'(?i)\b(?:phase|task|step|action)\s*(?:\d[\d-]*\s*)?:|[<>]',guide)
                    and not re.match(r'(?i)^(silent|standby|response)\b',guide)):
                return slots({'step':guide}), 'explicit-selected-response'
    return invalid, 'unresolved'
