#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


# ------------------ time + ASS helpers ------------------

def ass_color_from_rgb(hex_color: str) -> str:
    # "#RRGGBB" -> "&H00BBGGRR&"
    h = hex_color.strip()
    if h.startswith("#"):
        h = h[1:]
    h = h.lower()
    if not re.fullmatch(r"[0-9a-f]{6}", h):
        raise ValueError(f"Invalid color '{hex_color}' (expected #RRGGBB)")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}&"


def ass_ts(t: float) -> str:
    # ASS time: H:MM:SS.cc
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    t -= 3600 * h
    m = int(t // 60)
    t -= 60 * m
    s = int(t)
    cs = int(round((t - s) * 100))
    if cs == 100:
        s += 1
        cs = 0
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        h += 1
        m = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    text = text.replace("\\", r"\\")
    text = text.replace("{", r"\{").replace("}", r"\}")
    text = text.replace("\n", r"\N")
    return text


def srt_ts_to_seconds(ts: str) -> float:
    m = re.fullmatch(r"(\d\d):(\d\d):(\d\d),(\d\d\d)", ts.strip())
    if not m:
        raise ValueError(f"Bad SRT timestamp: {ts!r}")
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


# ------------------ parsing ------------------

def parse_srt(path: Path):
    """
    Returns list: [{start, end, text}, ...]
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    blocks = re.split(r"\n\s*\n", raw)
    out = []

    ts_re = re.compile(r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)")

    for b in blocks:
        lines = [ln.rstrip("\r") for ln in b.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        m = ts_re.search(lines[1])
        if not m:
            continue
        st = srt_ts_to_seconds(m.group(1))
        et = srt_ts_to_seconds(m.group(2))
        text = " ".join(lines[2:]).strip()
        if text:
            out.append({"start": st, "end": et, "text": text})
    return out


def load_words_from_json(path: Path):
    """
    Flat time-ordered list: [{word,start,end}, ...]
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = []
    for seg in payload.get("segments", []) or []:
        for w in (seg.get("words", []) or []):
            token = (w.get("word") or "").strip()
            if not token:
                continue
            words.append(
                {
                    "word": token,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                }
            )
    words.sort(key=lambda x: (x["start"], x["end"]))
    return words


# ------------------ alignment with merge/split ------------------

_norm_re = re.compile(r"[^\w]+", re.UNICODE)

def norm(tok: str) -> str:
    # Lowercase, strip punctuation-ish. Good enough for “compound vs split” cases.
    t = tok.strip().lower()
    t = _norm_re.sub("", t)
    return t


def tokenize_srt(text: str) -> list[str]:
    # Proofread text: preserve punctuation as attached tokens; we only space-split.
    return [t for t in text.split() if t.strip()]


def align_spans(srt_tokens: list[str], json_words: list[str]) -> list[tuple[int, int] | None]:
    """
    Align JSON words (timing slots) to SRT tokens (display tokens).
    Returns spans per JSON word: (i_start, i_end_excl) in SRT token indices, or None.

    Supports:
    - many JSON -> one SRT token  (compound joined in SRT)
      e.g. SRT "Hausnummer" vs JSON ["Haus", "nummer"]
    - one JSON -> many SRT tokens  (rare, but handle)
      e.g. SRT ["Haus", "nummer"] vs JSON ["Hausnummer"]

    Heuristic greedy with small lookahead. Works well if edits are mostly punctuation/spelling
    and occasional compounding changes.
    """
    a = srt_tokens
    b = json_words
    an = [norm(x) for x in a]
    bn = [norm(x) for x in b]

    spans: list[tuple[int, int] | None] = [None] * len(b)

    i = 0  # SRT token index
    j = 0  # JSON word index

    def is_empty_norm(x: str) -> bool:
        return x == ""

    while j < len(b) and i < len(a):
        # Skip SRT tokens that normalize to empty (pure punctuation etc.)
        if is_empty_norm(an[i]):
            i += 1
            continue
        # Skip JSON words that normalize to empty (shouldn't happen, but be safe)
        if is_empty_norm(bn[j]):
            spans[j] = None
            j += 1
            continue

        # 1:1 match
        if an[i] == bn[j]:
            spans[j] = (i, i + 1)
            i += 1
            j += 1
            continue

        # Many JSON -> one SRT (JSON split, SRT compound)
        # Try bn[j] + bn[j+1] + ... == an[i]
        matched = False
        acc = ""
        jj = j
        while jj < len(b) and len(acc) <= len(an[i]) and (jj - j) < 4:  # cap merge length
            if bn[jj]:
                acc += bn[jj]
            if acc == an[i]:
                # Map all these JSON words to same SRT token
                for k in range(j, jj + 1):
                    spans[k] = (i, i + 1)
                i += 1
                j = jj + 1
                matched = True
                break
            jj += 1
        if matched:
            continue

        # One JSON -> many SRT (SRT split, JSON compound)
        # Try an[i] + an[i+1] + ... == bn[j]
        acc = ""
        ii = i
        while ii < len(a) and len(acc) <= len(bn[j]) and (ii - i) < 4:  # cap split length
            if an[ii]:
                acc += an[ii]
            if acc == bn[j]:
                spans[j] = (i, ii + 1)  # highlight across multiple SRT tokens
                i = ii + 1
                j += 1
                matched = True
                break
            ii += 1
        if matched:
            continue

        # Fallback: advance whichever side seems “ahead”.
        # If the current SRT token norm is a substring of the JSON word norm, assume SRT is split -> advance i.
        if an[i] and bn[j] and an[i] in bn[j]:
            i += 1
        else:
            # Otherwise assume JSON has extra split/noise -> advance j.
            spans[j] = (i, i + 1)  # keep something highlightable instead of None
            j += 1

    # Fill remaining JSON with last known span (stability > perfection)
    last = None
    for idx in range(len(spans)):
        if spans[idx] is None:
            spans[idx] = last
        else:
            last = spans[idx]

    # If everything was None, leave it (caller will handle)
    return spans


def render_line_with_span(
    srt_tokens: list[str],
    span: tuple[int, int] | None,
    *,
    primary_ass: str,
    hilite_ass: str,
) -> str:
    if not srt_tokens:
        return ""

    lo, hi = (-1, -1)
    if span is not None:
        lo, hi = span

    out = []
    for idx, tok in enumerate(srt_tokens):
        esc = ass_escape(tok)
        if span is not None and lo <= idx < hi:
            out.append(rf"{{\c{hilite_ass}}}{esc}{{\c{primary_ass}}}")
        else:
            out.append(esc)
    return " ".join(out).strip()


# ------------------ ASS output ------------------

def make_header(*, res_x: int, res_y: int, font: str, font_size: int,
                primary_ass: str, outline: int, shadow: int, align: int) -> str:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {res_x}\n"
        f"PlayResY: {res_y}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{font_size},{primary_ass},{primary_ass},&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,{outline},{shadow},{align},0,0,0,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def main():
    ap = argparse.ArgumentParser(
        description="ASS word-highlight from proofread SRT text + faster-whisper JSON timing grid (robust to compounds)."
    )
    ap.add_argument("stem", help="Reads stem.srt + stem.json; writes stem.ass unless --out")
    ap.add_argument("--out", default=None)

    ap.add_argument("--res", required=True, help="WxH, e.g. 1080x1920")
    ap.add_argument("--font", required=True)
    ap.add_argument("--font-size", type=int, required=True)
    ap.add_argument("--primary", required=True, help="#RRGGBB")
    ap.add_argument("--highlight", required=True, help="#RRGGBB")
    ap.add_argument("--outline", type=int, default=3)
    ap.add_argument("--shadow", type=int, default=0)

    ap.add_argument("--align", type=int, default=6, help="ASS alignment 1..9 (default 6 middle-right)")
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--y", type=int, required=True)

    ap.add_argument("--slop", type=float, default=0.12, help="Seconds: include JSON words near SRT edges")
    ap.add_argument("--min-word-dur", type=float, default=0.10, help="Seconds: clamp ultra-short highlights")

    args = ap.parse_args()

    stem = Path(args.stem)
    srt_path = stem.with_suffix(".srt")
    json_path = stem.with_suffix(".json")
    out_ass = Path(args.out).expanduser().resolve() if args.out else stem.with_suffix(".ass")

    if not srt_path.exists():
        raise SystemExit(f"Missing: {srt_path}")
    if not json_path.exists():
        raise SystemExit(f"Missing: {json_path}")

    w_str, h_str = args.res.lower().split("x", 1)
    res_x, res_y = int(w_str), int(h_str)

    if not (1 <= args.align <= 9):
        raise SystemExit("--align must be 1..9")

    primary_ass = ass_color_from_rgb(args.primary)
    hilite_ass = ass_color_from_rgb(args.highlight)

    blocks = parse_srt(srt_path)
    grid_words = load_words_from_json(json_path)
    if not blocks:
        raise SystemExit("No parsable SRT blocks.")
    if not grid_words:
        raise SystemExit("JSON contains no words[]. Did you transcribe with word_timestamps=True?")

    header = make_header(
        res_x=res_x,
        res_y=res_y,
        font=args.font,
        font_size=args.font_size,
        primary_ass=primary_ass,
        outline=args.outline,
        shadow=args.shadow,
        align=args.align,
    )

    tag_prefix = rf"{{\an{args.align}\pos({args.x},{args.y})\c{primary_ass}}}"

    # Pointer into global JSON word list for efficient window scans
    j0 = 0
    events = []

    for blk in blocks:
        st = float(blk["start"])
        et = float(blk["end"])
        srt_text = blk["text"]
        srt_tokens = tokenize_srt(srt_text)
        if not srt_tokens:
            continue

        # Collect JSON words whose start falls into this block window (+slop)
        a = st - args.slop
        b = et + args.slop

        while j0 < len(grid_words) and grid_words[j0]["start"] < a:
            j0 += 1

        k = j0
        local = []
        while k < len(grid_words) and grid_words[k]["start"] <= b:
            local.append(grid_words[k])
            k += 1

        if not local:
            continue

        local_json_tokens = [w["word"] for w in local]
        spans = align_spans(srt_tokens, local_json_tokens)

        # Emit one Dialogue per JSON word slot
        for idx, w in enumerate(local):
            ws = max(st, float(w["start"]))

            if idx + 1 < len(local):
                we = min(et, float(local[idx + 1]["start"]))
            else:
                we = min(et, float(w["end"]))

            if we - ws < args.min_word_dur:
                we = min(et, ws + args.min_word_dur)

            if we <= ws:
                continue

            line = render_line_with_span(
                srt_tokens,
                spans[idx],
                primary_ass=primary_ass,
                hilite_ass=hilite_ass,
            )
            events.append((ws, we, line))

    if not events:
        raise SystemExit("No ASS events generated. Increase --slop or check SRT/JSON overlap.")

    with open(out_ass, "w", encoding="utf-8") as f:
        f.write(header)
        for st, et, line in events:
            f.write(f"Dialogue: 0,{ass_ts(st)},{ass_ts(et)},Default,,0,0,0,,{tag_prefix}{line}\n")

    print(f"Wrote {out_ass}")


if __name__ == "__main__":
    main()
