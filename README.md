# ProactiveCoach

**[Improving Proactive AI Assistance with Hierarchical Procedural Understanding](https://arxiv.org/abs/2610.06505)**

[Jin-Seop Lee](https://jinsuby.github.io/)<sup>1</sup>, [TaeYeon Won](https://www.linkedin.com/in/taeyeon3073)<sup>1</sup>, [SeongJun Jung](https://www.linkedin.com/in/seongjun-jung-3939223b2)<sup>1</sup>, JungHoon Kim<sup>1</sup>, [Boyang Albert Li](http://www.boyangli.org/index.html)<sup>2</sup>, JinYeong Bak<sup>1</sup>, [Jaehong Yoon](https://jaehong31.github.io/)<sup>2,&#42;</sup>, [Jee-Hyong Lee](https://iislab.skku.edu/members/)<sup>1,&#42;</sup>

<sup>1</sup> Department of Artificial Intelligence, Sungkyunkwan University, Republic of Korea  
<sup>2</sup> College of Computing and Data Science, Nanyang Technological University, Singapore  
<sup>&#42;</sup> Corresponding authors

[Paper (PDF)](https://arxiv.org/pdf/2610.06505) · [arXiv:2610.06505](https://arxiv.org/abs/2610.06505) · [BibTeX](#citation)

[Project page](https://jinsuby.github.io/ProactiveCoach/) · [Video examples](https://jinsuby.github.io/ProactiveCoach/#examples) · [Dataset preview](https://anonymous.4open.science/r/dataset-review-7c3e/)

ProactiveCoach connects **phases, steps, and actions** to help an assistant decide what to say, when to respond, and how much detail to provide during a procedural task.

## Citation

```bibtex
@misc{proactive,
  title = {Improving Proactive AI Assistance with Hierarchical Procedural Understanding},
  author = {Lee, Jin-Seop and Won, TaeYeon and Jung, SeongJun and Kim, JungHoon and Li, Boyang Albert and Bak, JinYeong and Yoon, Jaehong and Lee, Jee-Hyong},
  year = {2026},
  note = {Preprint},
  eprint = {2610.06505},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.06505}
}
```

## Highlights

- **Hierarchical guidance:** joint training across three procedural levels, with a lightweight router that adapts the displayed guidance to user requests.
- **Data and evaluation:** ProactiveCoach-Instruct contains 10,007 training samples; ProactiveCoachBench contains 1,112 evaluation samples.
- **Interactive examples:** five HoloAssist videos with synchronized phase, step, and action annotations. Each video plays from the beginning. The [annotation preview](site/examples.json) contains 30 cases.

## Results

Hierarchical supervision improves step-level Avg. across four backbones on ProactiveCoachBench (paper, Table 3). Gains are absolute score points.

| Backbone | Step-only | Hierarchical | Gain |
| --- | ---: | ---: | ---: |
| Qwen3-VL-4B | 49.32 | 58.96 | +9.64 |
| Qwen3-VL-8B | 53.07 | 59.37 | +6.30 |
| Qwen3.5-4B | 50.06 | 59.30 | +9.24 |
| Qwen3.5-9B | 51.18 | 58.81 | +7.63 |

See the [project page](https://jinsuby.github.io/ProactiveCoach/#results) for multi-level results and metric definitions.

## Getting started

Training, inference, and routing require a compatible CUDA environment. Start with Python 3.12 and provision the packages listed in [Dependencies](docs/DEPENDENCIES.md). A locked environment, full training data, benchmark anchors, and trained checkpoints are not included.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python tools/verify_source.py
python train.py --help
```

Prepare videos and JSONL records using the [data formats](docs/DATA_FORMATS.md), and supply a compatible model checkpoint. The paths below are examples.

### Training

```bash
export TRAIN_JSONL=data/train.jsonl
export VIDEO_ROOT=data/videos
export OUTPUT_DIR=outputs/proactivecoach
export MODEL=Qwen/Qwen3.5-4B
export MODEL_TYPE=qwen35
export NPROC_PER_NODE=1
bash scripts/train.sh
```

For Qwen3-VL, use `MODEL_TYPE=qwen3vl` with a compatible `MODEL`. Set `CUDA_VISIBLE_DEVICES` for your allocated GPUs and use a new output directory. **Before training, resolve the supplied `warmup_steps=0.05` setting**; see [training notes](docs/RELEASE_NOTES.md#review-before-experiments-or-publication).

### Inference and evaluation

```bash
mkdir -p outputs
python infer.py --model checkpoints/proactivecoach --model-type qwen35 \
  --data data/eval.jsonl --video-root data/videos \
  --anchors data/anchors.jsonl --output outputs/predictions.jsonl --history gt

python evaluate.py --data data/eval.jsonl --anchors data/anchors.jsonl \
  --predictions outputs/predictions.jsonl --output outputs/decision-scores.json \
  --skip-semantic --export-judge-requests outputs/judge-requests.jsonl
```

Benchmark evaluation requires ground-truth history and matching prediction/anchor coverage. Generated-history inference is a demo mode. The command above produces decision scores; full semantic and PQS scoring additionally require the semantic model and judge outputs. API judging sends evaluation content to the configured external service and may incur charges.

## Code guide

| Entry point | Purpose |
| --- | --- |
| `train.py`, `scripts/train.sh` | Streaming supervised fine-tuning |
| `infer.py` | Anchored inference and generated-history demos |
| `evaluate.py` | Decision, semantic, and PQS evaluation |
| `route.py` | Guidance-level routing over saved predictions |
| `scripts/judge_pqs.py` | API judging for PQS |
| `human_eval/aggregate.py` | Human-rating aggregation |

Each entry point provides `--help`. See [data formats](docs/DATA_FORMATS.md), [dependencies](docs/DEPENDENCIES.md), and the [file inventory](FILE_INVENTORY.md) for details.

## License and acknowledgments

See [LICENSE](LICENSE) for the code license and third-party notices. Example videos are from [HoloAssist](https://holoassist.github.io/) under [CDLA-Permissive 2.0](site/licenses/CDLA-Permissive-2.0.txt); see [video credits](docs/VIDEO_SAMPLE.md). External models and datasets retain their own licenses.

