# Dependency inventory

Derived from supplied imports and configuration; this is not a tested requirements file. Exact versions, source revisions, CUDA/driver versions, and transitive dependencies remain to be supplied. No dependency setup or GPU workload was run.

| Workflow | Observed dependencies/interfaces |
| --- | --- |
| Training and video inference | `torch`, `transformers`, `numpy`, `qwen-vl-utils` (`qwen_vl_utils`), PyAV (`av`) |
| Training patches | `liger-kernel` (`liger_kernel`), PyTorch FlexAttention, private Dynamo/Inductor interfaces |
| Vision training attention | `flash_attention_2` selected in `train.py`; compatible FlashAttention installation needed |
| Distributed launcher | `torchrun`, DeepSpeed through `--deepspeed scripts/zero1.json` |
| Trainer | Compatible Transformers Trainer runtime and integration dependencies; versions absent |
| Semantic scoring | `sentence-transformers`, `numpy`, `scipy`; default all-mpnet-base-v2 encoder |
| Router | `torch`, `transformers`, CUDA streams, supplied causal model and tokenizer |
| Decision-only scoring, PQS HTTP driver, human aggregation | Standard library and included modules; judge API access for uncached requests |

Private/version-sensitive interfaces include Liger loss/output classes, Transformers attention/masking registries, Qwen3.5 and Qwen3-VL model classes, and qwen-vl-utils video-reader internals. `infer.py` imports both Qwen model classes unconditionally. Successful syntax checks do not validate those imports or interfaces.

The package initializer overwrites seven recipe environment variables. It selects sparse guidance, two-second chunks, legacy-preserving behavior, and a PyAV reader registered under the torchvision backend name. That backend name alone does not establish a pinned torchvision dependency; resolve transitive requirements against the authors' working environment.

For reproducibility, obtain the original environment lock, model/processor revisions, CUDA/GPU details, and a validated small end-to-end run. Python 3.12.10 source parsing is the only Python compatibility claim made here.
