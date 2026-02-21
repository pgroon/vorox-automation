#!/home/groon/Vorox/07_Videos/tools/whisper/.venv/bin/python

import argparse
import json
import os
from pathlib import Path

from faster_whisper import WhisperModel


def srt_ts(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    t -= 3600 * h
    m = int(t // 60)
    t -= 60 * m
    s = int(t)
    ms = int(round((t - s) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_audio", help="wav/mp3/flac/etc.")
    ap.add_argument("--model-dir", default="/home/groon/LLM/faster-whisper-small")
    ap.add_argument("--language", default="de")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    audio_path = Path(args.input_audio).expanduser().resolve()
    base = audio_path.with_suffix("")  # myaudio.wav -> myaudio
    out_srt = base.with_suffix(".srt")
    out_json = base.with_suffix(".json")

    model_dir = os.path.expanduser(args.model_dir)
    model = WhisperModel(model_dir, device=args.device, compute_type=args.compute_type)

    print("Using language:", args.language)
    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=True,
        initial_prompt="Literarische Lesung eines klassischen deutschen Textes.",
    )

    segments = list(segments)

    # 1) Write "normal" SRT: one subtitle per segment (no chunking)
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            text = (seg.text or "").strip()
            if not text:
                continue
            f.write(f"{i}\n{srt_ts(seg.start)} --> {srt_ts(seg.end)}\n{text}\n\n")

    print(
        f"Wrote {out_srt} (detected_language={getattr(info,'language',None)}, "
        f"prob={getattr(info,'language_probability',None)})"
    )

    # 2) Write JSON with segments + per-word timestamps
    payload = {
        "audio": str(audio_path),
        "meta": {
            "model_dir": model_dir,
            "language": args.language,
            "detected_language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "beam_size": args.beam_size,
            "device": args.device,
            "compute_type": args.compute_type,
        },
        "segments": [],
    }

    for seg in segments:
        seg_obj = {
            "id": int(getattr(seg, "id", len(payload["segments"]))),
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
            "words": [],
        }

        words = getattr(seg, "words", None) or []
        for w in words:
            token = (w.word or "").strip()
            if not token:
                continue
            seg_obj["words"].append(
                {
                    "word": token,
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": float(w.probability) if getattr(w, "probability", None) is not None else None,
                }
            )

        payload["segments"].append(seg_obj)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
