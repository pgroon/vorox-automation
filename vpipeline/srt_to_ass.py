#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path


# ------------------ time + ASS helpers ------------------

def ass_color_from_rgb(hex_color: str) -> str:
    # "#RRGGBB" -> "&H00BBGGRR&" (ASS uses BBGGRR)
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

def align_spans(srt_tokens: list[str], json_words: list[str]) -> list[tuple[int, int] | None]:
    """
    Alignment policy:
      1) If non-empty token counts match: map positionally 1:1. (No fuzzy checks.)
      2) Else try limited merge/split based on exact normalized concatenation.
      3) Else fall back to positional mapping over the shorter side (best-effort),
         leaving extras mapped to nearest token so chunking doesn't wedge.
    """
    an = [norm(t) for t in srt_tokens]
    bn = [norm(w) for w in json_words]

    srt_idx = [i for i, t in enumerate(an) if t != ""]
    json_idx = [j for j, w in enumerate(bn) if w != ""]

    spans: list[tuple[int, int] | None] = [None] * len(json_words)

    # -------- 1) Fast path: counts match -> positional mapping ----------
    if len(srt_idx) == len(json_idx):
        for k in range(len(json_idx)):
            j = json_idx[k]
            i = srt_idx[k]
            spans[j] = (i, i + 1)
        return spans

    # -------- 2) Merge/split reconciliation (exact concat, no fuzzy) ----
    i_ptr = 0  # pointer into srt_idx
    j_ptr = 0  # pointer into json_idx

    while j_ptr < len(json_idx) and i_ptr < len(srt_idx):
        i = srt_idx[i_ptr]
        j = json_idx[j_ptr]

        # exact 1:1 normalized
        if an[i] == bn[j]:
            spans[j] = (i, i + 1)
            i_ptr += 1
            j_ptr += 1
            continue

        matched = False

        # many JSON -> one SRT (JSON split, SRT compound / hyphenated)
        # bn[j] + bn[j+1] + ... == an[i]
        acc = ""
        jj = j_ptr
        while jj < len(json_idx) and (jj - j_ptr) < 6 and len(acc) <= len(an[i]):
            acc += bn[json_idx[jj]]
            if acc == an[i] and acc != "":
                for k in range(j_ptr, jj + 1):
                    spans[json_idx[k]] = (i, i + 1)
                i_ptr += 1
                j_ptr = jj + 1
                matched = True
                break
            jj += 1
        if matched:
            continue

        # one JSON -> many SRT (SRT split, JSON compound)
        # an[i] + an[i+1] + ... == bn[j]
        acc = ""
        ii = i_ptr
        while ii < len(srt_idx) and (ii - i_ptr) < 6 and len(acc) <= len(bn[j]):
            acc += an[srt_idx[ii]]
            if acc == bn[j] and acc != "":
                spans[j] = (srt_idx[i_ptr], srt_idx[ii] + 1)
                i_ptr = ii + 1
                j_ptr += 1
                matched = True
                break
            ii += 1
        if matched:
            continue

        # -------- 3) Best-effort fallback: accept substitution ----------
        # Whisper got the word wrong, you corrected it. We don't care about similarity.
        spans[j] = (i, i + 1)
        i_ptr += 1
        j_ptr += 1

    # Map any remaining JSON words to the last available SRT token (prevents "dropped tail")
    last_i = srt_idx[-1] if srt_idx else None
    while j_ptr < len(json_idx):
        j = json_idx[j_ptr]
        if last_i is not None:
            spans[j] = (last_i, last_i + 1)
        j_ptr += 1

    # If SRT has extra tokens, we ignore them; they still show in display because chunk slicing uses tok_idx.

    return spans

# ------------------ parsing ------------------
def parse_srt(path: Path):
    """Returns list: [{start, end, text}, ...]"""
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
    """Flat time-ordered list: [{word,start,end}, ...]"""
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
    # Lowercase, strip punctuation-ish. Good enough for German compound join/split.
    t = tok.strip().lower()
    t = _norm_re.sub("", t)
    return t

def tokenize_srt(text: str) -> list[str]:
    return [t for t in text.split() if t.strip()]

# ------------------ rendering ------------------

def render_line_with_span_subset(
    srt_tokens: list[str],
    subset_idx: list[int],
    span: tuple[int, int] | None,
    *,
    primary_ass: str,
    hilite_ass: str,
) -> str:
    """
    Render only tokens referenced by subset_idx, keeping their original order.
    Highlight the intersection of subset_idx with the given span.
    Inserts a 2-line wrap using \\N when chunk length > 3 (up to 6).
    """
    if not subset_idx:
        return ""

    lo, hi = (-1, -1)
    if span is not None:
        lo, hi = span

    # Build raw token list for this chunk (used for choosing wrap point)
    raw_tokens = []
    for ti in subset_idx:
        if 0 <= ti < len(srt_tokens):
            raw_tokens.append(srt_tokens[ti])

    if not raw_tokens:
        return ""

    split_k = choose_two_line_split(raw_tokens, max_words=6)  # <-- your rule

    # Now build escaped/highlighted output in the same order
    out_tokens = []
    for ti in subset_idx:
        if not (0 <= ti < len(srt_tokens)):
            continue
        tok = srt_tokens[ti]
        esc = ass_escape(tok)
        if span is not None and lo <= ti < hi:
            out_tokens.append(rf"{{\c{hilite_ass}}}{esc}{{\c{primary_ass}}}")
        else:
            out_tokens.append(esc)

    if not out_tokens:
        return ""

    # Apply the split by token count (NOT by character count after tagging)
    if split_k is None or split_k <= 0 or split_k >= len(out_tokens):
        return " ".join(out_tokens).strip()

    line1 = " ".join(out_tokens[:split_k]).strip()
    line2 = " ".join(out_tokens[split_k:]).strip()
    return (line1 + r"\N" + line2).strip()


def choose_two_line_split(tokens: list[str], max_words: int = 6) -> int | None:
    """
    Return k (1..n-1) meaning: put tokens[:k] on line 1, tokens[k:] on line 2.
    Return None => single line.
    """
    n = len(tokens)
    if n <= 3:
        return None

    if n > max_words:
        n = max_words
        tokens = tokens[:n]

    best_k = 3  # default
    best_score: int | None = None

    for k in range(1, n):
        left = " ".join(tokens[:k])
        right = " ".join(tokens[k:])
        score = abs(len(left) - len(right))

        # avoid widows (1-word line)
        if k == 1 or k == n - 1:
            score += 5

        if best_score is None or score < best_score:
            best_score = score
            best_k = k

    return best_k

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
        description="Generate ASS word-highlight from proofread SRT (text truth) + faster-whisper JSON (timing truth)."
    )
    ap.add_argument("stem", help="Reads stem.srt + stem.json; writes stem.ass unless --out")
    ap.add_argument("--out", default=None)

    # video/layout
    ap.add_argument("--res", required=True, help="WxH, e.g. 1080x1920")
    ap.add_argument("--align", type=int, default=6, help="ASS alignment 1..9 (default 6 middle-right)")
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--y", type=int, required=True)

    # style
    ap.add_argument("--font", required=True)
    ap.add_argument("--font-size", type=int, required=True)
    ap.add_argument("--primary", required=True, help="#RRGGBB")
    ap.add_argument("--highlight", required=True, help="#RRGGBB")
    ap.add_argument("--outline", type=int, default=3)
    ap.add_argument("--shadow", type=int, default=0)

    # timing behavior
    ap.add_argument("--slop", type=float, default=0.12, help="Seconds: include JSON words near SRT edges")
    ap.add_argument("--min-word-dur", type=float, default=0.10, help="Seconds: clamp ultra-short highlights")

    # chunking (display only)
    ap.add_argument("--max-words", type=int, default=0, help="If >0, cap chunk length in (unique) SRT tokens")
    ap.add_argument("--min-words", type=int, default=2, help="Minimum words before punctuation split")
    ap.add_argument("--pad", type=float, default=0.15, help="End padding in seconds (clamped to SRT block end)")

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

    # pointer into global JSON word list for efficient window scan
    j0 = 0
    events: list[tuple[float, float, str]] = []

    punct_set = set(",.;:!?")

    for blk in blocks:
        st = float(blk["start"])
        et = float(blk["end"])
        srt_tokens = tokenize_srt(blk["text"])
        if not srt_tokens:
            continue

        # collect JSON words within this SRT block window (+slop)
        a = st - float(args.slop)
        b = et + float(args.slop)

        while j0 < len(grid_words) and grid_words[j0]["start"] < a:
            j0 += 1

        k = j0
        local = []
        while k < len(grid_words) and grid_words[k]["start"] <= b:
            local.append(grid_words[k])
            k += 1

        if not local:
            # nothing timed under this SRT block; skip (or warn later if you want)
            continue

        local_json_tokens = [w["word"] for w in local]
        spans = align_spans(srt_tokens, local_json_tokens)

        # Warn if alignment quality is poor for this block.
        unmapped = sum(1 for sp in spans if sp is None)
        if unmapped:
            frac = unmapped / max(1, len(spans))
            if frac >= 0.25:
                print(
                    f"[warn] weak alignment in block {ass_ts(st)}-{ass_ts(et)}: {unmapped}/{len(spans)} unmapped",
                    file=sys.stderr,
                )

        # Chunk the DISPLAY tokens based on SRT tokens, but time it on the JSON grid.
        # We chunk by accumulating unique SRT token indices referenced by consecutive JSON word spans.
        chunks = []
        buf_json: list[int] = []
        buf_tok_idx: list[int] = []
        seen_tok: set[int] = set()
        last_span_for_tokens: tuple[int, int] | None = None

        def push_chunk() -> None:
            nonlocal buf_json, buf_tok_idx, seen_tok
            if not buf_json:
                return
            j_first = buf_json[0]
            j_last = buf_json[-1]
            ch_start = float(local[j_first]["start"])
            # clamp padded end to SRT block end
            ch_end = min(float(local[j_last]["end"]) + float(args.pad), et)
            chunks.append({
                "j_first": j_first,
                "j_last": j_last,
                "start": ch_start,
                "end": ch_end,
                "tok_idx": list(buf_tok_idx),
            })
            buf_json = []
            buf_tok_idx = []
            seen_tok = set()

        for j_rel in range(len(local)):
            spn = spans[j_rel]
            spn_for_tokens = spn if spn is not None else last_span_for_tokens
            if spn_for_tokens is None:
                # Nothing to display yet; keep the word in timing grid but we can't add tokens.
                buf_json.append(j_rel)
                continue

            # remember last usable span for token accumulation
            last_span_for_tokens = spn_for_tokens

            lo, hi = spn_for_tokens
            # add unique SRT token indices for this span
            for ti in range(lo, hi):
                if 0 <= ti < len(srt_tokens) and ti not in seen_tok:
                    seen_tok.add(ti)
                    buf_tok_idx.append(ti)

            buf_json.append(j_rel)

            word_count = len(buf_tok_idx)

            # punctuation boundary based on the last token of this span
            last_tok = srt_tokens[hi - 1] if (hi - 1) < len(srt_tokens) else ""
            punct_break = bool(last_tok and last_tok[-1] in punct_set and word_count >= int(args.min_words))

            cap_break = bool(int(args.max_words) > 0 and word_count >= int(args.max_words))

            # Critical: do NOT split between JSON words that map to the same SRT token span
            # (e.g. when you merge a German compound in the proofread SRT: "weg-von-hier").
            # Otherwise the next chunk may start with a JSON word whose highlight span is
            # not present in that chunk's token subset, causing the highlight to "stick".
            same_span_as_next = False
            if (punct_break or cap_break) and (j_rel + 1) < len(local):
                cur = spans[j_rel]
                nxt = spans[j_rel + 1]
                if cur is not None and nxt is not None and cur == nxt:
                    same_span_as_next = True

            if (punct_break or cap_break) and not same_span_as_next:
                push_chunk()

        push_chunk()  # tail

        # Emit word-highlight events within each chunk
        for ch in chunks:
            j_first = ch["j_first"]
            j_last = ch["j_last"]
            ch_start = float(ch["start"])
            ch_end = float(ch["end"])
            tok_idx = ch["tok_idx"]

            for j_rel in range(j_first, j_last + 1):
                w = local[j_rel]
                ws = max(ch_start, float(w["start"]))

                if j_rel + 1 <= j_last:
                    we = min(ch_end, float(local[j_rel + 1]["start"]))
                else:
                    we = min(ch_end, float(w["end"]))

                if we - ws < float(args.min_word_dur):
                    we = min(ch_end, ws + float(args.min_word_dur))

                if we <= ws:
                    continue

                line = render_line_with_span_subset(
                    srt_tokens,
                    tok_idx,
                    spans[j_rel],
                    primary_ass=primary_ass,
                    hilite_ass=hilite_ass,
                )
                if not line:
                    continue

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
