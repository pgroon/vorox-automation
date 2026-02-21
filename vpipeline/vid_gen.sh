#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./vid_gen.sh background.png audio.wav [highlight_color]
#
# Options:
#   --color HEX     highlight color for visualizer (RRGGBB / #RRGGBB / 0xRRGGBB)
#   --ass  FILE     burn ASS subtitles (optional; styling comes from the ASS file)
#   --vert          vertical output (1080x1920). Default is horizontal (2560x1440)

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <background.png> <audio.wav> [highlight_color] [--color HEX] [--ass FILE] [--vert]"
  exit 1
fi

BG="$1"
AUDIO="$2"

# Defaults
RAW_COLOR="d79921"   # highlight for visualizer
ASSFILE=""           # optional subtitle burn-in
VERT=0

# Convert "#RRGGBB" / "0xRRGGBB" / "RRGGBB" -> "RRGGBB"
norm_hex6() {
  local h="$1"
  h="${h#\#}"
  h="${h#0x}"
  echo "$h"
}

# Parse args after BG + AUDIO
shift 2
while [ "$#" -gt 0 ]; do
  case "$1" in
    --vert)
      VERT=1
      ;;
    --color)
      shift
      RAW_COLOR="${1:-d79921}"
      ;;
    --ass)
      shift
      ASSFILE="${1:-}"
      ;;
    *)
      # Allow bare highlight color as positional arg
      if [[ "$1" =~ ^#?[0-9A-Fa-f]{6}$ ]] || [[ "$1" =~ ^0x[0-9A-Fa-f]{6}$ ]]; then
        RAW_COLOR="$1"
      else
        echo "Unknown argument: $1"
        exit 1
      fi
      ;;
  esac
  shift
done

# Normalize highlight color to 0xRRGGBB for showfreqs
RAW_COLOR="$(norm_hex6 "$RAW_COLOR")"
COLOR="0x${RAW_COLOR}"

# Validate ASS file if provided
if [ -n "$ASSFILE" ] && [ ! -f "$ASSFILE" ]; then
  echo "ASS not found: $ASSFILE"
  exit 1
fi

# Derive output name from audio file: input.wav -> input.mp4
BASE="${AUDIO%.*}"

# Canvas sizes and other vars depending on format
if [ "$VERT" -eq 1 ]; then
  OUT_W=1080
  OUT_H=1920
  STRIP_W=320
  OUT="${BASE}_vert.mp4"
else
  OUT_W=2560
  OUT_H=1440
  STRIP_W=700
  OUT="${BASE}.mp4"
fi

BARS=40
GAP=4
PERIOD=$(( OUT_H / BARS ))
KEEP=$(( PERIOD - GAP ))

# Optional ASS burn-in (single encode)
SUB_STAGE=""
MAP_V="[v0]"
if [ -n "$ASSFILE" ]; then
  # libass reads the style from the ASS file; no force_style here.
  SUB_STAGE=";[v0]subtitles='${ASSFILE}'[v]"
  MAP_V="[v]"
fi

ffmpeg \
  -loop 1 -i "$BG" \
  -i "$AUDIO" \
  -filter_complex "\
        [0:v]scale=${OUT_W}:${OUT_H},setsar=1,format=rgba[bg];\
        [1:a]showfreqs=s=$((BARS+1))x${STRIP_W}:mode=bar:ascale=log:fscale=log:win_size=8192:overlap=1:colors=${COLOR},\
        crop=${BARS}:${STRIP_W}:0:0,\
        transpose=1,\
        scale=${STRIP_W}:${OUT_H}:flags=neighbor,\
        format=rgba,\
        geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lt(mod(Y,${PERIOD}),${KEEP}),alpha(X,Y),0)'[spec];\
        [bg][spec]overlay=x=0:y=0:format=auto:shortest=1[v0]\
${SUB_STAGE}" \
  -map "$MAP_V" -map 1:a \
  -c:v libx264 -preset slow -crf 18 \
  -c:a aac -b:a 192k \
  -shortest \
  "$OUT"

echo "--------------------------------------------------------"
echo "VERT=$VERT OUT_W=$OUT_W OUT_H=$OUT_H STRIP_W=$STRIP_W Color=${COLOR}"
if [ -n "$ASSFILE" ]; then
  echo "ASSFILE=$ASSFILE"
fi
echo "--------------------------------------------------------"
