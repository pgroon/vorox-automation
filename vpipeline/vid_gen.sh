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

unset ASSFILE
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


#----------------------------------------------------------------------------------------------------------
#                                   FFMPEG magic - Some notes:
#----------------------------------------------------------------------------------------------------------

# ffmpeg invocation to generate a static-background video with a left-aligned
# frequency-bar visualizer strip and optional subtitle stage.
#
# Inputs:
#   -loop 1 -i "$BG"
#       Loop the background image indefinitely (turn single image into a video stream).
#   -i "$AUDIO"
#       Main audio input (drives both sound and visualization).
#
# Filter graph (-filter_complex):
#
#   [0:v]scale=${OUT_W}:${OUT_H},setsar=1,format=rgba[bg];
#       - Scale background image to final output resolution.
#       - setsar=1 forces square pixels (avoid inherited weird SAR).
#       - Convert to RGBA to ensure alpha-capable compositing.
#       - Label result as [bg].
#
#   [1:a]showfreqs=...
#       Generate frequency spectrum from audio:
#         s=$((BARS+1))x${STRIP_W}
#             Internal resolution of spectrum (freq bins × strip width).
#             +1 bin avoids right-edge clipping after crop.
#         mode=bar
#             Render vertical bars (not line/combined).
#         ascale=log
#             Logarithmic amplitude scaling (more perceptually useful).
#         fscale=log
#             Logarithmic frequency spacing (denser low-end detail).
#         win_size=8192
#             FFT window size (higher = better frequency resolution, slower response).
#         overlap=1
#             Maximum frame overlap (smoother animation, more CPU).
#         colors=${COLOR}
#             Bar color definition.
#
#       crop=${BARS}:${STRIP_W}:0:0
#           Trim to exact number of bars (remove extra helper column).
#
#       transpose=1
#           Rotate spectrum (so bars become vertical strip layout as intended).
#
#       scale=${STRIP_W}:${OUT_H}:flags=neighbor
#           Stretch spectrum to full output height.
#           neighbor scaling keeps hard pixel edges (no smoothing/blur).
#
#       format=rgba
#           Ensure alpha channel for later masking.
#
#       geq=... a='if(lt(mod(Y,${PERIOD}),${KEEP}),alpha(X,Y),0)'
#           Per-pixel alpha mask:
#             - Create horizontal stripe pattern.
#             - Only keep rows where (Y mod PERIOD) < KEEP.
#             - Produces dashed / segmented bar look.
#           Result labeled as [spec].
#
#   [bg][spec]overlay=x=0:y=0:format=auto:shortest=1[v0]
#       Overlay spectrum strip onto background at top-left (0,0).
#       format=auto lets ffmpeg negotiate pixel format.
#       shortest=1 stops overlay when shortest input ends.
#       Output labeled as [v0].
#
#   ${SUB_STAGE}
#       Optional subtitle filter chain appended here
#       (e.g., subtitles=..., ass=..., etc.).
#
# Stream mapping:
#   -map "$MAP_V"
#       Select final filtered video stream (e.g. [v0] or subtitle-processed label).
#   -map 1:a
#       Map original audio stream from second input.
#
# Encoding:
#   -c:v libx264
#       H.264 video codec.
#   -preset slow
#       Better compression efficiency (more CPU time).
#   -crf 18
#       High quality visually lossless-ish setting (lower = higher quality).
#
#   -c:a aac -b:a 192k
#       AAC audio at 192 kbps.
#
#   -shortest
#       End output when shortest stream ends (prevents infinite background loop).
#
#   "$OUT"
#       Final output file.

# Optional ASS burn-in (vertical only)
SUB_STAGE=""
MAP_V="[v0]"

if [ "$VERT" -eq 1 ] && [ -n "${ASSFILE:-}" ]; then
  # Escape single quotes for ffmpeg/libass: '  ->  \'
  ASS_ESC=${ASSFILE//\'/\\\'}
  SUB_STAGE=";[v0]subtitles='${ASS_ESC}'[v]"
  MAP_V="[v]"
fi

FILTERGRAPH=$(
  cat <<EOF
[0:v]scale=${OUT_W}:${OUT_H},setsar=1,format=rgba[bg];
[1:a]showfreqs=s=$((BARS+1))x${STRIP_W}:mode=bar:ascale=log:fscale=log:win_size=8192:overlap=1:colors=${COLOR},
crop=${BARS}:${STRIP_W}:0:0,
transpose=1,
scale=${STRIP_W}:${OUT_H}:flags=neighbor,
format=rgba,
geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lt(mod(Y,${PERIOD}),${KEEP}),alpha(X,Y),0)'[spec];
[bg][spec]overlay=x=0:y=0:format=auto:shortest=1[v0]${SUB_STAGE}
EOF
)

ffmpeg \
  -loop 1 -i "$BG" \
  -i "$AUDIO" \
  -filter_complex "$FILTERGRAPH" \
  -map "$MAP_V" -map 1:a \
  -c:v libx264 -preset slow -crf 18 \
  -c:a aac -b:a 192k \
  -shortest \
  "$OUT"

echo "--------------------------------------------------------"
echo "VERT=$VERT OUT_W=$OUT_W OUT_H=$OUT_H STRIP_W=$STRIP_W Color=${COLOR}"
if [ "$VERT" -eq 1 ] && [ -n "$ASSFILE" ]; then
  echo "Subtitles:=$ASSFILE"
fi
echo "--------------------------------------------------------"