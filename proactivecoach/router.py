from concurrent.futures import ThreadPoolExecutor
import time

ROUTER_SYSTEM = 'Select the guidance granularity requested by the user.\nPhase = a broad intermediate goal containing multiple procedure steps.\nStep = one coherent operation with a recognizable result, containing smaller actions.\nAction = one small executable manipulation or movement.\nThe current display level is provided. Interpret the new request relative to it when needed.\nMore detail means a finer level; less detail means a coarser level. If the user names the\ndesired scope explicitly, use it. If no change is requested, keep the current level.\nReply with exactly one word: Phase, Step, or Action. Do not provide the guidance itself.\nExamples:\nCurrent: Action. Request: Only tell me the broad stages. Answer: Phase\nCurrent: Phase. Request: Use intermediate operations, not broad stages or tiny motions. Answer: Step\nCurrent: Step. Request: Explain every individual hand movement. Answer: Action\nCurrent: Step. Request: Keep the current granularity. Answer: Step\n'
KEYS = ('Phase', 'Step', 'Action')

class AsyncRouter:

    def __init__(self, model_path, device='cuda:0', system=ROUTER_SYSTEM):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.device = device
        self.system = system
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='level-router')
        self.origin = time.perf_counter()
        self._run([('Step', 'Keep the current guidance level.')], time.perf_counter())

    def _run(self, items, submitted):
        started = time.perf_counter()
        with self.torch.inference_mode(), self.torch.cuda.stream(self.stream):
            prompts = [self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system}, {'role': 'user', 'content': f'Current display level: {level}\nNew user request: {request}'}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for level, request in items]
            batch = self.tokenizer(prompts, padding=True, return_tensors='pt').to(self.device)
            output = self.model.generate(**batch, do_sample=False, max_new_tokens=8, pad_token_id=self.tokenizer.pad_token_id)
            texts = self.tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
            self.stream.synchronize()
        completed = time.perf_counter()
        return {'outputs': [{'raw': text, 'level': text.strip() if text.strip() in KEYS else None} for text in texts], 'queue_ms': 1000 * (started - submitted), 'compute_ms': 1000 * (completed - started), 'latency_ms': 1000 * (completed - submitted), 'submitted_wall_offset_sec': submitted - self.origin, 'completed_wall_offset_sec': completed - self.origin}

    def submit(self, items):
        return self.pool.submit(self._run, items, time.perf_counter())

    def close(self):
        self.pool.shutdown(wait=True)


def select_guidance(raw, level):
    from evaluation.parser import extract
    if level not in KEYS:
        raise ValueError('Expected Phase, Step, or Action')
    slots, _ = extract(raw, 'joint', 'sparse')
    slot = slots[level.lower()]
    return dict(selected_level=level, decision=slot['decision'],
                text=slot['text'] if slot['decision']=='response' else '')


class DisplayRouter:
    def __init__(self, router, initial_level='Step'):
        if initial_level not in KEYS:
            raise ValueError('Invalid initial display level')
        self.router, self.level, self.pending = router, initial_level, None

    def request(self, text):
        if self.pending is not None:
            raise RuntimeError('A routing request is already pending')
        self.pending = self.router.submit([(self.level,text)])

    def publish(self, raw):
        if self.pending is not None and self.pending.done():
            level = self.pending.result()['outputs'][0]['level']
            if level in KEYS:
                self.level = level
            self.pending = None
        return select_guidance(raw,self.level)
