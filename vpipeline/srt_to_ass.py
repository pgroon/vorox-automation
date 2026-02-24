#!/usr/bin/env python3

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
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

def edit_distance_bounded(a: str, b: str, max_dist: int) -> int:
    """Return Levenshtein distance if <= max_dist, else max_dist+1 (early exit)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1

    # keep b the shorter string to reduce work
    if lb > la:
        a, b = b, a
        la, lb = lb, la

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution
            )
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > max_dist:
            return max_dist + 1
        prev = cur

    return prev[lb]


def token_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False

    la, lb = len(a), len(b)

    # Narrow short-token rule: allow only tiny edits + strong prefix constraint.
    # Fixes cases like "her" -> "herr" without making short words broadly fuzzy.
    if 3 <= la <= 5 and 3 <= lb <= 5 and a.isalpha() and b.isalpha():
        if a[:2] == b[:2] and edit_distance_bounded(a, b, 1) <= 1:
            return True

    # Allow very common short inflection endings when one token is a prefix of the other.
    # Fixes cases like "ihn" <-> "ihnen" without enabling broad fuzzy matching on short words.
    if a.isalpha() and b.isalpha():
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        if 3 <= len(short) <= 5 and 4 <= len(long) <= 7:
            if long.startswith(short):
                suffix = long[len(short):]
                if suffix in {"e", "en", "em", "er", "es", "n"}:
                    return True

                # Reject very short/ambiguous tokens (keeps "ihn" vs "ihnen" from silently matching)
    if min(la, lb) < 6:
        return False

    # Large length jumps are likely different words.
    if abs(la - lb) > 2:
        return False

    # 1) high similarity
    if SequenceMatcher(None, a, b).ratio() >= 0.90:
        return True

    # 2) tolerate small edit distance for long words (typos)
    if min(la, lb) >= 8 and a[0] == b[0] and edit_distance_bounded(a, b, 2) <= 2:
        return True

    return False

def align_spans(srt_tokens: list[str], json_words: list[str]) -> list[tuple[int, int] | None]:
    """
    Align JSON word slots to SRT token spans.

    Returns spans per JSON word: (i_start, i_end_excl) in SRT token indices, or None.

    Supports:
      - many JSON -> one SRT token  (compound joined in SRT)
      - one JSON -> many SRT tokens (rare)
      - conservative fuzzy substitutions (spelling corrections)

    Assumes:
      - edits are mostly punctuation/spelling + occasional compounding
      - SRT timestamps anchor alignment per block
    """
    a = srt_tokens
    b = json_words

    an = [norm(x) for x in a]
    bn = [norm(x) for x in b]

    spans: list[tuple[int, int] | None] = [None] * len(b)

    i = 0  # index in SRT
    j = 0  # index in JSON

    def empty(x: str) -> bool:
        return x == ""

    while j < len(b) and i < len(a):
        if empty(an[i]):
            i += 1
            continue

        if empty(bn[j]):
            spans[j] = None
            j += 1
            continue

        # ----------------------------
        # 1:1 match (exact or fuzzy)
        # ----------------------------
        if token_eq(an[i], bn[j]):
            spans[j] = (i, i + 1)
            i += 1
            j += 1
            continue

        matched = False

        # ---------------------------------------------------
        # many JSON -> one SRT (JSON split, SRT compound)
        # Example: JSON ["weg","von","hier"] -> SRT ["weg-von-hier"]
        # ---------------------------------------------------
        acc = ""
        jj = j
        while jj < len(b) and (jj - j) < 4:
            if bn[jj]:
                acc += bn[jj]
            if token_eq(acc, an[i]):
                for k in range(j, jj + 1):
                    spans[k] = (i, i + 1)
                i += 1
                j = jj + 1
                matched = True
                break
            jj += 1

        if matched:
            continue

        # ---------------------------------------------------
        # one JSON -> many SRT (SRT split, JSON compound)
        # Example: JSON ["hausnummer"] -> SRT ["haus","nummer"]
        # ---------------------------------------------------
        acc = ""
        ii = i
        while ii < len(a) and (ii - i) < 4:
            if an[ii]:
                acc += an[ii]
            if token_eq(acc, bn[j]):
                spans[j] = (i, ii + 1)
                i = ii + 1
                j += 1
                matched = True
                break
            ii += 1

        if matched:
            continue

        # ---------------------------------------------------
        # fallback: advance cautiously without hard-mapping
        # ---------------------------------------------------
        # If SRT token seems to be substring of JSON word,
        # likely SRT split; advance SRT.
        if an[i] and bn[j] and an[i] in bn[j]:
            i += 1
        else:
            # leave as None; move JSON forward
            spans[j] = None
            j += 1

    # -------------------------------------------------------
    # LIMITED forward-fill (avoid catastrophic drift)
    # Only tolerate up to 2 consecutive unmapped slots.
    # -------------------------------------------------------
    last = None
    run = 0

    for idx in range(len(spans)):
        if spans[idx] is None:
            if last is not None and run < 2:
                spans[idx] = last
                run += 1
            else:
                run += 1
        else:
            last = spans[idx]
            run = 0

    return spans

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