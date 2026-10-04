# Data contracts

Derived from supplied code. The archive contains no released data records.

## Video records (JSONL)

Each nonblank line is one object with `video_id`, `video_path`, `video_start`, `video_end`, `video_chunk_size`, `thoughts`, and `conversations`. IDs must be unique for inference/evaluation. Relative video paths resolve beneath `--video-root`; prefer relative paths in shared manifests.

Start/end are numeric seconds: finite, nonempty, nonnegative start, and 1–540 complete two-second chunks. `video_chunk_size` must be 2. `thoughts` contains exactly one target per chunk in chunk order; each has numeric `timestamp` within that chunk and `assistant` text. `think`, if present, must be empty.

`conversations` is nonempty and contains only `role: "user"` entries with numeric `timestamp` and `content`. All timestamps lie in the interval, with at least one instruction in the first chunk. Keep the task goal first: evaluation uses the first conversation's content as context.

Assistant text is exactly `<silent>`, or `<response> ` followed by entries in `phase`, `step`, `action` order. Entry forms are `phase 1: guidance`, `step 1-1: guidance`, `action 1-1-1: guidance`, separated by ` / `. IDs are positive; guidance is nonempty; duplicate level/ID pairs in one response are rejected. `proactivecoach/supervision.py::parse` defines the exact grammar.

## Anchors (JSONL)

Each anchor has unique `sample_id`, a `video_id` from video records, zero-based integer `chunk`, numeric `time = video_start + 2 * (chunk + 1)`, and a nonempty unique `levels` list selected from `phase`, `step`, `action`.

Evaluation rejects repeated video/chunk/level combinations, shifted times, prediction-coordinate mismatches, and incomplete or extra prediction IDs. Official anchor selection and split files are missing; do not label a newly constructed split as the published benchmark.

## Predictions (JSONL)

`infer.py` writes `sample_id`, `video_id`, `chunk`, `time`, `raw`, `levels`, `parse_route`, `frame_times`, and `history`. Evaluation reparses `raw`; cached `levels` are not authoritative. Benchmark scoring requires `history: "gt"`.

The predictor replays recent visual chunks with strictly past text history. Generated-history demos still require the loader's valid annotation schema, including thoughts, and dense anchors from chunk zero.

## Router requests (JSONL)

Required fields: `video_id`, numeric `time`, `request` text. Requests are sorted per video and applied before the first saved packet at or after their time. Requests after the last packet have no packet to apply to. CLI level labels are case-sensitive `Phase`, `Step`, `Action`.

## PQS requests/judgments

Export requests with `evaluate.py --export-judge-requests`; each contains `id`, `judge`, and `prompt`. IDs hash judge identity and prompt. The driver writes matching `id`, `judge`, and `scores`; four dimension names are defined in `evaluation/judge_prompt.py`, with numeric values 1–5. Missing required judgments yield null PQS. Duplicate IDs are rejected.

## Human ratings (CSV)

Headers: `case_id,reviewer_id,level,metric,score`. Levels: `phase`, `step`, `action`. Metrics: `alignment`, `coverage`, `relevance`, `quality`. Each case/reviewer pair must supply all 12 combinations, including missing ratings. Scores are integer strings 0–5, `U` for unratable, or blank for missing. The aggregator produces counts and means, not a human study. Use `human_eval/rubric.json` for definitions.
