# VOROX Audio - Narration Video Pipeline

This repo automates narration video generation for YouTube (long-form and Shorts) without a GUI editor.

## What you get

Given `audio.wav`:

- **Long-form:** a horizontal video with background + audio visualizer.
- **Shorts:** a vertical video with background + audio visualizer + burned-in ASS subtitles with word-by-word highlight.

## Files

### `transcribe.py`
Runs faster-whisper and produces:

- `audio.srt` — normal SRT for proofreading (punctuation/spelling fixes)
- `audio.json` — per-word timestamps (timing grid for highlight burn-in)

Usage:
```bash
./transcribe.py audio.wav
```

Important: when proofreading `audio.srt`, change **text only**. Do not edit timestamps unless you know what you're doing.

### `srt_to_ass.py`
Converts proofread `audio.srt` + `audio.json` into `audio.ass` (word highlight, hard-switch).
This script is called by `build.py` for vertical output.

### `build.py`
Interactive “hub” script:

- generates the background image (`audio.png`)
- generates ASS subtitles for vertical output
- calls `vid_gen.sh` to produce the final mp4

Usage:
```bash
./build.py
```

### `vid_gen.sh`
Single ffmpeg encode:

- background image loop
- audio visualizer
- optional ASS burn-in
- outputs:
  - `audio.mp4` (horizontal)
  - `audio_vert.mp4` (vertical)

## Typical workflow

1) Record `audio.wav`
2) Transcribe:
```bash
./transcribe.py audio.wav
```
3) Proofread `audio.srt` (text/punctuation/spelling)
4) Build video(s):
```bash
./build.py
```
Choose horizontal or vertical in the prompt.

