# Five procedure videos from the beginning

The project page shows H083, H086, H123, H154 and H081 from original time zero, for up to 180 seconds each. All five supplied sources are shorter than 180 seconds, so their full duration is used. Nothing is padded or replayed to reach three minutes. The original files are retained locally; the public page includes only web-compressed derivatives and first-frame posters.

The complete 30-case corrected `site/examples.json` is unchanged. `site/featured-examples.json` limits only the UI. These are ground-truth examples, not model predictions or evidence of model performance.

| Case | Task | Original length (s) | Web playback (s) | MP4 bytes |
| --- | --- | ---: | ---: | ---: |
| H083 | Printer paper and toner | 115.957021 | 116 | 6266646 |
| H086 | Tray-frame assembly | 168.020593 | 168.1 | 7101779 |
| H123 | Computer RAM replacement | 166.008439 | 166.1 | 10010858 |
| H154 | Coffee preparation | 130.011813 | 130.1 | 6892407 |
| H081 | Relay and fuse replacement | 130.014710 | 130.1 | 7588302 |

## Timing and preserved annotations

All viewing windows begin at 0 seconds. `original_review_time = playback_time`; the offset is zero. The viewer derives slider limits, timeline scales and reset state from the new media durations. At time zero, no guidance is shown until the first recorded guide timestamp. Event selection uses guide_time, or event start for a null guide. End-of-video replay resets to zero. Phase/step/action text, original spans, parent indices, null guidance and timestamps are unchanged. Downloading an annotation returns the complete original case. The viewer still supports earlier context if a future viewing window starts later, but none of these five windows requires pre-start context.

The source review videos have approximately 10 fps. The web derivatives use a 10 fps frame grid; output duration may differ by less than 0.1 seconds due to frame rounding. This is not a time offset or speed change. The last source frames are retained, and annotation timestamps are not stretched.

The specified 30-case candidate file matches the corrected public English annotations, treating null/empty guidance equivalently. Older annotations bundled in the video ZIP were not substituted. The older archive name HUPPA is historical; this project uses ProactiveCoach.

## Selection and permission

The five tasks cover printer maintenance, assembly, computer hardware, coffee preparation and electrical equipment maintenance. H083/H086/H123/H154 were in the supplied recommended list; H081 adds a distinct task from the same designated candidate file. H062/H064 were excluded for Ego-Exo4D redistribution restrictions. Other HoloAssist candidates repeat coffee, printer or assembly tasks; no claim is made that these are the statistically best five cases.

Source: supplied `ThinkStream_Offline_386_Full.zip`, members `ThinkStream_Offline_386/dataset/media/{ID}.mp4`. Source recording IDs, original hashes, derivative hashes and intervals are in `site/media.json`. Original files were verified unchanged after encoding.

The official HoloAssist Download section states CDLA-Permissive 2.0: https://holoassist.github.io/ . Its section 2.1 permits sharing modified or unmodified data with the agreement text: https://cdla.dev/permissive-2-0/ . The agreement accompanies these files at `site/licenses/CDLA-Permissive-2.0.txt`. The existing code MIT license is unchanged and does not replace the data license.

Citation: Xin Wang, Taein Kwon, et al. *HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World*. ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Wang_HoloAssist_An_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_in_ICCV_2023_paper.html

## Reproduce web compression

FFmpeg 9.0.2 essentials from Gyan was used. This distributor is linked under Windows EXE Files at https://ffmpeg.org/download.html . Distribution page: https://www.gyan.dev/ffmpeg/builds/ . The downloaded archive matched its published SHA-256: `60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba`. FFmpeg itself is not distributed in this repository.

For LENGTH, use the smaller of the original duration and 180 seconds:

```sh
ffmpeg -i original/ID.mp4 -t LENGTH -map 0:v:0 -an -map_metadata -1 -c:v libx264 -preset slow -crf 25 -maxrate 480k -bufsize 960k -pix_fmt yuv420p -r 10 -fps_mode cfr -threads 2 -movflags +faststart site/media/clips/ID.mp4
ffmpeg -i site/media/clips/ID.mp4 -frames:v 1 -q:v 3 site/media/clips/ID.jpg
```

Resolution remains 638 x 360, with normal-speed H.264 video, no audio, and fast-start metadata. A constrained bitrate keeps individual uploads within connector limits without lowering resolution. All source review videos were already silent. Versioned URLs avoid reusing the earlier short clips from browser caches.
