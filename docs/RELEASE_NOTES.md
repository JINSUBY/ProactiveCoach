# Release preparation

All 28 original files, including LICENSE, retain their paths and bytes. Added files provide documentation, hashes, static verification, and Git exclusions. Research code, prompts, defaults, and metrics were not changed. No Git repository was created or pushed.

## Missing reproduction inputs

- Training/evaluation annotations, videos, official splits, and official benchmark anchors.
- Dataset distribution locations and data-use terms. The reference manuscript's anonymous preview is not established here as a downloadable training/benchmark release.
- Trained checkpoints and processor/tokenizer revisions; router-model choice/revision and external model terms.
- Exact package/source versions, CUDA/GPU environment, validated launch settings, and a small verified end-to-end run.
- Final paper/project URL and author-approved citation.

## Review before experiments or publication

`train.py` passes `warmup_steps=0.05` to `TrainingArguments`. This is a fractional value in a parameter named for steps. Without the original dependency version and author intent, it is unclear whether this was intended as a 5% warmup ratio, fractional steps, or another setting. It may be rejected or interpreted unexpectedly by the selected runtime. No runtime behavior is asserted and no conversion to `warmup_ratio=0.05` or integer steps was made. Confirm the intended setting and validate it in the original environment before training.

Private model-patching interfaces require compatible PyTorch, Transformers, Liger, and qwen-vl-utils revisions. Static checks cannot establish that compatibility. `route.py --device cpu` is not a supported workaround: the router explicitly constructs CUDA streams. Package imports overwrite selected recipe environment settings; those side effects remain unchanged.

The original MIT license carries CASIA-IVA-Lab/ThinkStream attribution. Confirm its scope and attribution for the complete release. No new licensing choice was made.

Source naming is ProactiveCoach, matching the reference manuscript. No HUPPA occurrence was found. ThinkStream environment names and legacy-format terminology are preserved. The reference PDF contains draft editorial annotations and is excluded; no paper acceptance status is claimed.

## Security and portability

The source ZIP passed integrity, bounded-size, path traversal, duplicate/case-collision, and symlink checks before extraction. Pattern scans found no obvious embedded credential literals or user/machine-specific home/mount paths. These are limited checks, not a guarantee of arbitrary execution safety.

The judge reads `OPENAI_API_KEY` from the environment and explicitly sends prompts to an API. Model loaders can download remote model resources. Private transfer metadata, local runtime, paper PDF, and inspection workspace are excluded from the release archive.

See [VERIFICATION.md](VERIFICATION.md) for checks and limits. No training, inference, GPU kernels, external model downloads, paid judging, semantic scoring, or human study were run.
