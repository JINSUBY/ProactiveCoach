# Verification report

Release preparation checks performed on Python 3.12.10 on 2026-10-03.

| Check | Result |
| --- | --- |
| Input ZIP integrity and extraction safety | PASS: 28 files; 99,314 uncompressed bytes; CRC, path traversal, symlink, size, duplicate/case-collision checks |
| Original file preservation | PASS: all 28 SHA-256 hashes match supplied archive |
| Original Python syntax/compilation without execution | PASS: 24 files |
| Added verification utility syntax | PASS: 1 file |
| JSON parsing | PASS: 2 original files plus source manifest |
| CLI help | PASS: train, infer, evaluate, route, judge_pqs, human aggregation |
| Original-source pattern scans | No obvious credential literals, machine-specific home/mount paths, or HUPPA occurrences found |
| Markdown local file links | PASS: all targets resolve within release |
| Final archive | PASS: 37 explicitly selected files; original bytes preserved; CRC and membership verified |

Run `python tools/verify_source.py` to repeat original-source hash and release syntax/JSON checks. CLI-help checks exit before model loading; they do not establish dependency compatibility. Pattern scans are limited, not a comprehensive security audit.

No research dependency installation, training, inference, GPU kernels, semantic encoder execution, external model downloads, paid API calls, or human study was run. Bash launcher execution/syntax under Bash was not checked on this Windows executor. No experiment metrics, runtime compatibility, GPU-memory requirement, or successful reproduction is claimed.

The release ZIP excludes the input PDF, transfer metadata, helper/runtime files, bytecode, and private account data. It contains the unchanged supplied LICENSE. No repository was created or pushed.
