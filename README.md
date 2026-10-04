# ProactiveCoach: supplementary research code

**Project page:** [https://jinsuby.github.io/ProactiveCoach/](https://jinsuby.github.io/ProactiveCoach/)

Code accompanying **Improving Proactive AI Assistance with Hierarchical Procedural Understanding**.

This archive preserves all 28 supplied research files byte-for-byte and adds release documentation and a static verification utility. It includes streaming supervised fine-tuning, inference, guidance-level routing, and evaluation.

**Reproduction status:** training, inference, and paper results have not been reproduced during release preparation. Datasets, benchmark anchors, trained checkpoints, and a tested dependency lockfile are not included. See [release notes](docs/RELEASE_NOTES.md) for outstanding items.

## Entry points

| File | Purpose |
| --- | --- |
| `train.py` | Streaming SFT for Qwen3.5 or Qwen3-VL |
| `scripts/train.sh` | Bash/torchrun launcher with DeepSpeed ZeRO stage 1 |
| `infer.py` | Supplied-anchor inference with ground-truth history, or dense generated-history demos |
| `evaluate.py` | Per-level decision, semantic, and PQS scoring; judge-request export |
| `route.py` | Replay user requests over saved predictions to select guidance level |
| `scripts/judge_pqs.py` | Submit exported PQS prompts to the configured API judge |
| `human_eval/aggregate.py` | Aggregate human-rating CSVs using the supplied rubric |

`proactivecoach/` contains model, attention, video, collation, supervision, inference, and routing code. `evaluation/` contains alignment, parsing, metrics, and judge prompts. See [file inventory](FILE_INVENTORY.md).

## Environment

Run commands from this directory. Python 3.12.10 passed syntax and CLI-help checks; this is not a validated training environment. No package pins, CUDA version, or hardware specification were supplied. See [dependency inventory](docs/DEPENDENCIES.md); installing arbitrary current versions is not a verified reproduction recipe.

Create an isolated environment with an available Python 3.12 installation, then provision compatible dependencies from the authors' environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Training, inference, and routing require CUDA in the supplied implementation. Training uses BF16 and selects FlashAttention 2 for vision attention. No minimum GPU memory requirement is claimed.

Safe source-only checks, without training dependencies:

```bash
python tools/verify_source.py
python train.py --help
python infer.py --help
python evaluate.py --help
python route.py --help
python scripts/judge_pqs.py --help
python human_eval/aggregate.py --help
```

## Data and models

Supply videos, training/evaluation JSONL records, and an official benchmark-anchor JSONL according to [data contracts](docs/DATA_FORMATS.md). No downloader, annotation-conversion pipeline, official split files, dataset download link, or trained checkpoint link is present in the source archive.

The training default is `Qwen/Qwen3.5-4B`, exactly as supplied. Its availability and runtime compatibility were not verified. Inference requires a model/checkpoint and its processor files; routing requires a separate causal-language-model checkpoint chosen by the user. External models and datasets retain their own terms.

## Training

Once missing inputs and a compatible environment are available, the existing launcher accepts:

```bash
export TRAIN_JSONL=data/train.jsonl
export VIDEO_ROOT=data/videos
export OUTPUT_DIR=outputs/proactivecoach
export MODEL=Qwen/Qwen3.5-4B
export MODEL_TYPE=qwen35
export NPROC_PER_NODE=1
bash scripts/train.sh
```

These paths are placeholders. Select allocated GPUs with `CUDA_VISIBLE_DEVICES`. For Qwen3-VL, set `MODEL_TYPE=qwen3vl` and provide a compatible `MODEL`. Additional `train.py` arguments can follow the launcher. Use a new output directory.

Defaults include a five-chunk visual window, two-second chunks, 98,304 maximum tokens, microbatch size 2, gradient accumulation 4, three epochs, and learning rate 2e-5. These can be expensive; they are not hardware recommendations. The loader refuses to truncate overlength sequences.

**Review before training:** `train.py` passes `warmup_steps=0.05`. The name denotes steps, while the fractional value may indicate an intended ratio. Confirm the original library behavior and author intent; this release does not silently convert it to `warmup_ratio` or an integer. See [release notes](docs/RELEASE_NOTES.md).

## Inference and evaluation

The following are command templates, not completed experiment runs:

```bash
mkdir -p outputs
python infer.py --model checkpoints/proactivecoach --model-type qwen35 \
  --data data/eval.jsonl --video-root data/videos \
  --anchors data/anchors.jsonl --output outputs/predictions.jsonl --history gt

python evaluate.py --data data/eval.jsonl --anchors data/anchors.jsonl \
  --predictions outputs/predictions.jsonl --output outputs/decision-scores.json \
  --skip-semantic --export-judge-requests outputs/judge-requests.jsonl
```

Benchmark evaluation requires ground-truth history and exact prediction/anchor coverage. `--history generated` requires all chunks starting at zero for every selected video and is a demo mode; its predictions are rejected by benchmark evaluation.

`--skip-semantic` leaves sP/sR/sF1 null. Missing required judgments leave PQS null. Avg is null unless sF1 and PQS are both available. Null metrics are not reproduced results.

Optional paid judging sends reference guidance, predicted guidance, and task context to the API. Supply `OPENAI_API_KEY` securely in the environment before explicitly running:

```bash
python scripts/judge_pqs.py --requests outputs/judge-requests.jsonl \
  --output outputs/judgments.jsonl
python evaluate.py --data data/eval.jsonl --anchors data/anchors.jsonl \
  --predictions outputs/predictions.jsonl --judgments outputs/judgments.jsonl \
  --output outputs/full-scores.json
```

The source fixes the judge identity to `gpt-5.2`. Full semantic scoring loads `sentence-transformers/all-mpnet-base-v2` on CPU by default. Neither model availability nor API access was verified; no paid calls were made.

## Routing and human evaluation

```bash
python route.py --model checkpoints/router --predictions outputs/predictions.jsonl \
  --requests data/requests.jsonl --output outputs/routed.jsonl \
  --initial-level Step --device cuda:0
python human_eval/aggregate.py data/ratings.csv --output outputs/human-scores.json
```

Create output parent directories first. Most entry points require new output filenames; the judge driver appends/resumes by request ID. Routing replays saved packets and waits for each routing result; it is not a complete live UI.

## Naming, provenance, and license

The source and supplied manuscript use **ProactiveCoach**, with **ProactiveCoach-Instruct** and **ProactiveCoachBench** in the manuscript. No HUPPA occurrence was found in the original source. `THINKSTREAM_*` names and legacy parsing remain for implementation provenance and compatibility.

The supplied [LICENSE](LICENSE) includes an MIT license, a 2026 CASIA-IVA-Lab copyright notice, and ThinkStream/third-party attribution. It is unchanged. Maintainers should confirm that this notice and permission cover the complete release before publication. No license was invented or substituted.

No paper PDF, weights, dataset, results, or private account metadata are bundled. A final paper/project link and author-approved citation remain to be supplied. See [verification results](docs/VERIFICATION.md).
