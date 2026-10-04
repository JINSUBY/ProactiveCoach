# Five short video examples

The project page displays five short HoloAssist excerpts with corrected review-sample-v0.2 ground-truth annotations. The full 30-case `site/examples.json` is unchanged; `site/featured-examples.json` controls the visible selection. These clips illustrate annotation structure, not model predictions or comparative model performance.

## Selected excerpts

The following intervals are in the original supplied review clips, not an independently reconstructed raw-recording timeline. Each excerpt is contiguous, runs at normal speed, and contains no generated frames or model output. The full original videos remain outside this public release.

| Case | Original review interval | Playback length | Focus |
| --- | --- | --- | --- |
| H083 | 39–69 s | 30 s | Paper tray opening, paper loading and closure. |
| H086 | 124–164 s | 40 s | Final tightening followed by placement of the tray. |
| H123 | 92–132 s | 40 s | New RAM installation guidance and the installation action. |
| H154 | 18–45 s | 27 s | Tank replacement, capsule preparation and brewing activation. |
| H081 | 98–130 s | 32 s | Fuse replacement followed by panel closure. |

H083, H086, H123 and H154 were in the supplied recommended list. H081 adds a distinct equipment-maintenance task from the exact supplied 30-case candidate file. Selection considers task diversity, hierarchy clarity, sampled visual alignment and redistribution terms. H062/H064 were excluded for Ego-Exo4D redistribution restrictions. Other HoloAssist candidates repeat coffee, printer or frame-assembly tasks; H144 also runs 570 seconds. Other source datasets were not selected because further public-video permission was not established. This is an editorial selection, not a claim of the statistically best cases.

## Timing and context

`original_review_time = excerpt_playback_time + excerpt_start`.

The media catalog supplies `excerpt_start`, `excerpt_end` and `video_time_offset` (equal to excerpt_start). The UI keeps original annotation timestamps internally. Playback controls, the seek slider and timeline ticks display excerpt-relative time; a second label shows the original review time. Timeline event bars are clipped only for display. A guidance marker is shown only when its original timestamp lies in the excerpt's half-open interval [start, end). Event selection seeks to its guide_time, or event start for a null guide, then subtracts the offset and clamps to the excerpt. Original event spans and guidance times remain visible in the inspector.

The latest prior guidance at each level is carried into the excerpt start and explicitly labeled **Earlier context**. Clicking such context inspects its original annotation and stays at excerpt time zero; it does not imply the instruction was issued again. Subsequent guidance updates use the original timestamp. Phase filtering respects original parent indices. Null guides are preserved. Downloading an annotation returns the complete original case, not a destructively trimmed record.

The supplied candidate file matches all 30 public cases' hierarchy, labels, English guidance and timestamps, treating empty/null guidance equivalently. Older annotation text in the media ZIP was not used to overwrite corrected public data. The ZIP uses older HUPPA naming; the public project retains ProactiveCoach.

## Source and license

Source media: supplied `ThinkStream_Offline_386_Full.zip`, members `ThinkStream_Offline_386/dataset/media/{ID}.mp4`. Original recording IDs, original-video hashes, derivative hashes and source intervals are recorded in `site/media.json`. These source review clips are silent and approximately 10 fps.

The official HoloAssist Download section states CDLA-Permissive 2.0: https://holoassist.github.io/ . Section 2.1 permits sharing modified or unmodified data when the agreement accompanies it: https://cdla.dev/permissive-2-0/ . Full agreement: `site/licenses/CDLA-Permissive-2.0.txt`. The code's original MIT license is unchanged and does not replace this data license.

Citation: Xin Wang, Taein Kwon, et al. *HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World*. ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Wang_HoloAssist_An_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_in_ICCV_2023_paper.html

## Reproduce the web derivatives

FFmpeg 9.0.2 essentials from Gyan was used. The distributor is linked by https://ffmpeg.org/download.html under Windows EXE Files. Download: https://www.gyan.dev/ffmpeg/builds/ . The downloaded archive SHA-256 matched the distributor's published checksum: `60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba`. The executable is not part of this repository.

For each original `{ID}.mp4`, substitute START and LENGTH from the table:

```sh
ffmpeg -ss START -i original/ID.mp4 -t LENGTH -map 0:v:0 -an -map_metadata -1 -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -r 10 -fps_mode cfr -threads 2 -movflags +faststart site/media/clips/ID.mp4
ffmpeg -i site/media/clips/ID.mp4 -frames:v 1 -q:v 3 site/media/clips/ID.jpg
```

Output is H.264 MP4, 638 x 360, 10 fps, without audio, with fast-start metadata. The transformation is a temporal crop and lossy recompression; it does not accelerate playback or change annotations. Frames are quantized to a 100 ms video grid, while guidance retains its exact original timestamps. Original source files were verified unchanged after conversion.

## Media hashes

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `site/media/clips/H081.jpg` | 27156 | `f0d69bd5a7429358abf3d2254b7a1b37f9335d04d67a4f3de6b34f2c75e691a3` |
| `site/media/clips/H081.mp4` | 3506759 | `be8bf13b2b0a84093712420b55c3a8e141673333b3ae83f0b053448246e09c4d` |
| `site/media/clips/H083.jpg` | 21720 | `4596e390c05e34bf2fba5198c37b88d224824698a8b4c7d7387d3eede597e95c` |
| `site/media/clips/H083.mp4` | 2091113 | `64d368aaa52a4ad554ff33e696a7ec5a93121ecd96d081dd3125a359752f4867` |
| `site/media/clips/H086.jpg` | 32475 | `ef4fcbca7509b9e4812c595c12bc58d29d9e283086bd4b7805c017f925db90c0` |
| `site/media/clips/H086.mp4` | 2443009 | `03d9e6d72632f83dbe6a5cdb2299d2d97aeda68eab643b1a4193827ce4d80d43` |
| `site/media/clips/H123.jpg` | 28544 | `7d37f3602c53df8c1e7a1ad8b24fc632be7a9c177ea6de74056dbdc08abbf485` |
| `site/media/clips/H123.mp4` | 5399609 | `2d57cb56794c5d29d6ed04dd30b793c38cc1b35162a2c24f082694e935521a01` |
| `site/media/clips/H154.jpg` | 27080 | `c9b82adf4832311466c94eb72438deda8735c4d6cd52711269773451c34a107e` |
| `site/media/clips/H154.mp4` | 1955327 | `5f70865618c5ea3b068721be05a843ea1f023765db278342ae364f052fe910b9` |
