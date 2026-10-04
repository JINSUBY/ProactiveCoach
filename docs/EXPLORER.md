# Actual-data explorer

The project page includes all 30 records from review-sample-v0.2 of the approved anonymous dataset preview. `site/examples.json` preserves each annotation record and appends its matching video metadata under `video`. Record IDs, original ordering, hierarchical parent indices, text, start/end, guide, and guide_time are unchanged.

Annotation-source SHA-256: `81abfacbabec1a73ff6f2268fdf5bfa8a7528ae0301273eb5427fca4dbcec5ab`.
Video-metadata SHA-256: `bd73da902bf25b7c1053acf82013193b45e8afccb8f35fcb0b319816186df9ef`.

Source preview: https://anonymous.4open.science/r/dataset-review-7c3e/

## Timing behavior

Guidance markers use guide_time, not event start. Bars use start/end. Marker and list selection seek to the exact recorded guidance time, or event start for a null guide. Equal-time guidance at each level is shown together. Latest guidance is explicitly held for inspection, not interpreted as continuous model speech. Null guides are not model silence. Arrays may be nonchronological; display sorting does not change source data. Phase filtering follows parent indices. Changing cases pauses and resets the annotation clock.

## Missing media

No usable media URL or original video is included in the preview metadata. `site/media-requirements.json` lists every required record, source dataset, clip interval, duration, and suggested clip filename. To enable a real video panel, supply publicly reusable clips or stable HTTPS URLs, confirm source terms and permissions, and confirm that video time zero matches annotation clip time zero. For multi-camera datasets, specify the intended camera. Source record suffixes and source timeline mapping still have TODOs in the supplied data dictionary; do not guess them.

No third-party video is fetched or redistributed by this explorer. It uses annotation-only playback and clearly identifies ground truth. Existing research code and manuscript metrics remain unchanged.
