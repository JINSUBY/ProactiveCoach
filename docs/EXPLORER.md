# Actual-data explorer

The project page displays five short video excerpts: H083, H086, H123, H154 and H081. The full 30-record review-sample-v0.2 dataset remains intact in `site/examples.json`, with original annotation fields and matching video metadata under `video`. `site/featured-examples.json` limits the UI without deleting source records. Selection rationale, source-video hashes and license details are in [VIDEO_SAMPLE.md](VIDEO_SAMPLE.md).

Annotation-source SHA-256: `81abfacbabec1a73ff6f2268fdf5bfa8a7528ae0301273eb5427fca4dbcec5ab`.
Video-metadata SHA-256: `bd73da902bf25b7c1053acf82013193b45e8afccb8f35fcb0b319816186df9ef`.
Source preview: https://anonymous.4open.science/r/dataset-review-7c3e/

## Timing behavior

Guidance markers use guide_time, not event start. Bars use start/end. Selection seeks to exact guidance time, or event start for a null guide. Equal-time guidance at each level is shown together. Latest guidance is held for inspection, not continuous model speech. Null guides are not model silence. Display sorting does not modify source arrays. Phase filtering follows parent indices. Changing examples pauses and resets playback. Native video controls and the annotation timeline share an excerpt clock; original-review time equals playback time plus excerpt_start. Original timestamps remain in the inspector. Prior phase/step guidance is retained as explicitly labeled earlier context. See VIDEO_SAMPLE.md for exact intervals and boundary behavior.

## Media

Five short, compressed derivatives of licensed HoloAssist review clips are bundled with CDLA-Permissive 2.0 text and attribution. These are 27–40 second excerpts at normal speed, encoded at 10 fps and silent. Only the derivatives are bundled; the original source clips are preserved outside the public repository. All existing corrected annotations are preserved. Other records remain available in the complete JSON file, but are not in the five-example UI. `site/media-requirements.json` retains the original full inventory and records which clips are available. Adding further videos requires confirmed source terms and matching clip time zero; source record suffixes and multi-camera mappings must not be guessed.

No model outputs or synthetic videos were added. Existing research code and manuscript metrics are unchanged.
