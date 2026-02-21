#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import traceback
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Presets for video background image layout
LAYOUTS = {
    "horizontal": {
        "orientation": "horizontal",
        "w": 2560,
        "h": 1440,
        "title_y": 240.0,
        "title_size": 160.0,
        "author_size": 86.0,
        "logo_h": 220.0,
        "logo_y": 1300,
        "margin": 180.0,
        "title_line_height": None,  # Defaults to (title_size * 1.15)
        "title_author_gap": None,  # Defaults to (title line height * 0.3)
    },
    "vertical": {
        "orientation": "vertical",
        "w": 1080,
        "h": 1920,
        "title_y": 350.0,
        "title_size": 100.0,
        "author_size": 66.0,
        "logo_h": 180.0,
        "logo_y": 1500,
        "margin": 60.0,
        "title_line_height": None,  # Defaults to (title_size * 1.15)
        "title_author_gap": None,  # Defaults to (title line height * 0.3)
    },
}

SCHEMES = {
    "EN originals": {"bg": "#282828", "text": "#928374", "highlight": "#fb4934"}, # tomato red 
    "DE originals":  {"bg": "#282828", "text": "#458588", "highlight": "#d79921"}, # yellow, this is the default!
    "EN classics":  {"bg": "#282828", "text": "#458588", "highlight": "#8f3f71"}, # purple
    "DE classics": {"bg": "#282828", "text": "#458588", "highlight": "#d65d0e"}, # burnt orange
    "EN poetry":    {"bg": "#282828", "text": "#458588", "highlight": "#689d6a"}, # aqua
    "DE poetry":    {"bg": "#282828", "text": "#458588", "highlight": "#98971a"}, # soft green
    "EN various":  {"bg": "#282828", "text": "#458588", "highlight": "#8a4f3d"}, # brick
    "DE various":  {"bg": "#282828", "text": "#458588", "highlight": "#6b7f3f"}, # moss green
}

DEFAULT_TITLE = ""
DEFAULT_AUTHOR = ""
KEEP_INTERMEDIATE_SVG = False  # keep out.svg for debugging
LOGO_SVG_PATH = Path("/home/groon/Vorox/07_Videos/assets/vorox_varlogo.svg")

SCRIPT_DIR = Path(__file__).resolve().parent

# Subtitle presets (burned into video for vertical / Shorts)
# Keep everything that affects typography/layout here so build.py is the single source of truth.
SUB_PRESETS = {
    "vertical": {
        # chunk/timing behaviour for srt_to_ass.py
        "slop": 0.12,
        "min_word_dur": 0.10,

        # style
        "font": "Oswald",
        "font_size": 52,
        "primary": "#FFFFFF",   # normal text color (do NOT reuse scheme['text'] blindly; readability first)
        # highlight color is set at runtime from the selected scheme by default (can be changed below)

        "outline": 3,
        "shadow": 0,

        # positioning (absolute pixel coordinates using \pos)
        "align": 6,     # middle-right
        "margin_r": 120,
        "y_offset": 0,  # add/subtract pixels from vertical center
    }
}


# ---------------------------------------------------------------------------
# SVG plumbing
# ---------------------------------------------------------------------------

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def prompt_user_inputs() -> dict:
    """
    Terminal prompts (no CLI args). Returns a small dict with user choices.
    """
    from InquirerPy import inquirer

    # Offer wav files in current directory as convenience
    wavs = sorted([p.name for p in Path.cwd().glob("*.wav")])
    audio_default = wavs[0] if wavs else ""
    srt_default = str(Path(audio_default).with_suffix(".srt")) if audio_default else ""

    layout = inquirer.select(
        message="Layout:",
        choices=[
            {"name": "Horizontal (2560x1440)", "value": "horizontal"},
            {"name": "Vertical (1080x1920)", "value": "vertical"},
        ],
        default="horizontal",
    ).execute()

    scheme = inquirer.select(
        message="Color scheme:",
        choices=list(SCHEMES.keys()),
        default="DE classics" if "EN originals" in SCHEMES else list(SCHEMES.keys())[0],
    ).execute()

    title = inquirer.text(
        message="Title:",
        default=DEFAULT_TITLE,
        validate=lambda s: len(s.strip()) > 0,
    ).execute()

    author = inquirer.text(
        message="Author:",
        default=DEFAULT_AUTHOR,
        validate=lambda s: len(s.strip()) > 0,
    ).execute()

    audio = inquirer.text(
        message="Audio file (.wav):",
        default=audio_default,
        validate=lambda s: Path(s).expanduser().exists(),
    ).execute()
    
    srt = inquirer.text(
        message="Subtitle file (.srt) [required for vertical]:",
        default=srt_default,
        validate=lambda s: (s.strip() == "") or Path(s).expanduser().exists(),
    ).execute()

    return {"layout": layout, "scheme": scheme, "title": title, "author": author, "audio": audio, "srt": srt}


def derive_outputs_from_audio(audio_path: Path) -> tuple[Path, Path]:
    # This checks the name of the audio and names the PNG accordingly.
    stem = audio_path.stem  # "another_kafka"
    png = Path.cwd() / f"{stem}.png"
    svg = png.with_suffix(".svg")
    return svg, png


def load_logo_as_group(logo_path: Path) -> tuple[ET.Element, float, float]:
    """
    Parse the logo SVG and return:
      - a <g> that contains all its children
      - the viewBox width/height for scaling
    """

    print(f"[logo] Using logo file: {logo_path}")
    if not logo_path.exists():
        raise FileNotFoundError(f"Logo SVG not found: {logo_path}")

    tree = ET.parse(logo_path)  # can raise ParseError if file isn't valid XML/SVG
    root = tree.getroot()

    vb = root.get("viewBox")
    if not vb:
        raise ValueError("Logo SVG is missing viewBox. Needed for reliable scaling.")

    # viewBox = "minX minY width height"
    _, _, vbw, vbh = map(float, vb.split())

    logo_g = ET.Element(q("g"), {"id": "vorox_logo"})
    for child in list(root):
        logo_g.append(child)

    return logo_g, vbw, vbh


def recolor_logo_groups(logo_g: ET.Element, text_color: str, highlight_color: str) -> None:
    """
    Recolor specific logo groups by their IDs:
      - logo-text and logo-sub   -> text_color
      - logo-accent              -> highlight_color

    Note: hardcoded 'fill' attributes in the SVG might override this.
    """
    for el in logo_g.iter():
        el_id = el.get("id")
        if el_id in ("logo-text", "logo-sub"):
            el.set("fill", text_color)
        elif el_id == "logo-accent":
            el.set("fill", highlight_color)


def normalize_lines(s: str) -> list[str]:
    # Convert typed escape sequences to real newlines.
    # handle "\\n" (two backslashes) before "\n" (one backslash).
    s = s.replace("\\\\n", "\n")
    s = s.replace("\\n", "\n")
    s = s.replace("|", "\n")

    lines = s.split("\n")
    return lines if lines else [""]


def count_lines(text: str) -> int:
    return len(normalize_lines(text))


def split_manual_lines(s: str) -> list[str]:
    """
    Supports:
      - literal '\n' typed in prompt (two chars)  -> line break
      - real newlines (if they ever occur)        -> line break
      - '|' as manual line break marker           -> line break
    """
    s = s.replace("\n", "\\n")   # make typed \n work
    s = s.replace("|", "\\n")     # make every | a break
    lines = s.split("\\n")        # split ALL breaks, not just first
    return lines if lines else [""]


def add_right_aligned_text(
    svg_root: ET.Element,
    *,
    text_id: str,
    content: str,
    x: float,
    y: float,
    font_size: float,
    fill: str,
    highlight_fill: str,
    line_height: float | None = None,   # px between lines; default derived from font_size
) -> None:
    """
    Right-aligned text that supports:
      - manual line breaks using '\n' or '|'
      - optional highlight segments inside [brackets]

    Brackets are not rendered; bracketed content is rendered with highlight_fill.
    """

    if line_height is None:
        line_height = font_size * 1.15  # sane default for Oswald

    text_el = ET.SubElement(svg_root, q("text"), {
        "id": text_id,
        "x": f"{x:.2f}",
        "y": f"{y:.2f}",
        "text-anchor": "end",
        "font-family": "Oswald",
        "font-weight": "300",
        "font-size": f"{font_size}px",
        "xml:space": "preserve",
    })

    # Split into lines: support either real newlines or '|' as a manual break marker
    lines = normalize_lines(content)

    # Helper: Add colored segments to a parent tspan (line container)
    def add_segments(parent: ET.Element, line: str) -> None:
        # Fast path: no brackets → one segment
        if "[" not in line or "]" not in line:
            seg = ET.SubElement(parent, q("tspan"), {"fill": fill})
            seg.text = line
            return

        parts = re.split(r"(\[[^\]]+\])", line)
        for part in parts:
            if not part:
                continue
            if part.startswith("[") and part.endswith("]"):
                seg = ET.SubElement(parent, q("tspan"), {"fill": highlight_fill})
                seg.text = part[1:-1]
            else:
                seg = ET.SubElement(parent, q("tspan"), {"fill": fill})
                seg.text = part

    # Create one <tspan> per line
    for i, line in enumerate(lines):
        attrs = {"x": f"{x:.2f}"}
        # dy moves down relative to previous line; first line uses dy=0
        attrs["dy"] = "0" if i == 0 else f"{line_height:.2f}"
        line_tspan = ET.SubElement(text_el, q("tspan"), attrs)
        add_segments(line_tspan, line)


def build_svg(
    *,
    out_svg_path: Path,
    width: int,
    height: int,
    bg: str,
    margin: float,
    logo_h: float,
    logo_y: float,
    text_color: str,
    highlight_color: str,
    title_text: str,
    author_text: str,
    title_size: float,
    author_size: float,
    title_y: float,
    title_line_height: float,
    title_author_gap: float,
) -> None:
    print(f"[svg] Building SVG: {out_svg_path}")
    # Create SVG and set viewBox
    svg_root = ET.Element(q("svg"), {
        "viewBox": f"0 0 {width} {height}",
        "width": str(width),
        "height": str(height),
    })
    
    # Draw background
    ET.SubElement(svg_root, q("rect"), {
        "x": "0", "y": "0",
        "width": str(width), "height": str(height),
        "fill": bg,
    })

    right_x = width - margin

    # Add story title
    add_right_aligned_text(
        svg_root,
        text_id="title_text",
        content=title_text,
        x=right_x,
        y=title_y,
        font_size=title_size,
        fill=text_color,
        highlight_fill=highlight_color,
    )

    # Add author name. First, calculate Y position of the author line:
    title_lh = title_line_height if title_line_height is not None else title_size * 1.15
    title_lines = count_lines(title_text)

    gap = title_lh * 0.3
    author_y = title_y + (title_lines * title_lh) + gap

    # Add author name:
    add_right_aligned_text(
        svg_root,
        text_id="author_text",
        content=author_text,
        x=right_x,
        y=author_y,
        font_size=author_size,
        fill=text_color,
        highlight_fill=highlight_color,
    )

    # Add logo
    logo_g, vbw, vbh = load_logo_as_group(LOGO_SVG_PATH)
    recolor_logo_groups(logo_g, text_color=text_color, highlight_color=highlight_color)

    scale = logo_h / vbh
    tx = (width - margin) - (vbw * scale)
    ty = (logo_y) - (vbh * scale)

    wrapper = ET.SubElement(svg_root, q("g"), {
        "transform": f"translate({tx:.2f},{ty:.2f}) scale({scale:.6f})"
    })
    wrapper.append(logo_g)

    ET.ElementTree(svg_root).write(out_svg_path, encoding="utf-8", xml_declaration=True)
    print(f"[svg] Wrote: {out_svg_path} ({out_svg_path.stat().st_size} bytes)")


def export_png_with_inkscape(svg_path: Path, png_path: Path) -> None:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        raise RuntimeError("Inkscape not found in PATH.")
    cmd = [
        inkscape,
        str(svg_path),
        "--export-type=png",
        f"--export-filename={png_path}",
    ]
    subprocess.run(cmd, check=True)
    print(f"[png] Wrote: {png_path} ({png_path.stat().st_size} bytes)")




def run_srt_to_ass(
    *,
    stem: Path,
    out_ass: Path,
    res_w: int,
    res_h: int,
    preset: dict,
    highlight_color: str,
) -> None:
    """Generate an ASS file from proofread SRT + word-timestamp JSON."""
    srt_to_ass = SCRIPT_DIR / "srt_to_ass.py"

    x = res_w - int(preset["margin_r"])
    y = (res_h // 2) + int(preset.get("y_offset", 0))

    cmd = [
        str(srt_to_ass),
        str(stem),
        "--out", str(out_ass),
        "--res", f"{res_w}x{res_h}",
        "--font", str(preset["font"]),
        "--font-size", str(int(preset["font_size"])),
        "--primary", str(preset["primary"]),
        "--highlight", str(highlight_color),
        "--outline", str(int(preset["outline"])),
        "--shadow", str(int(preset["shadow"])),
        "--align", str(int(preset["align"])),
        "--x", str(int(x)),
        "--y", str(int(y)),
        "--slop", str(float(preset["slop"])),
        "--min-word-dur", str(float(preset["min_word_dur"])),
    ]

    subprocess.run(cmd, check=True)

def run_vid_gen(
    image_path: Path,
    audio_path: Path,
    *,
    highlight_color: str,
    orientation: str,
    ass_path: Path | None,
) -> None:
    """Call vid_gen.sh with a stable interface."""
    vid_gen = SCRIPT_DIR / "vid_gen.sh"
    cmd = [
        str(vid_gen),
        str(image_path),
        str(audio_path),
        "--color",
        highlight_color,     # visualizer color
    ]

    if ass_path is not None:
        cmd += ["--ass", str(ass_path)]

    if orientation == "vertical":
        cmd.append("--vert")

    subprocess.run(cmd, check=True)


def main() -> None:
    conf = prompt_user_inputs()
    print(f"[pick] conf.layout = {conf['layout']!r}")
    layout = LAYOUTS[conf["layout"]]
    print(f"[layout] orientation={layout['orientation']!r} size={layout['w']}x{layout['h']}")
    scheme = SCHEMES[conf["scheme"]]

    audio_path = Path(conf["audio"]).expanduser().resolve()
    srt_raw = conf.get("srt", "").strip()
    srt_path = Path(srt_raw).expanduser().resolve() if srt_raw else None
    out_svg, out_png = derive_outputs_from_audio(audio_path)

    build_svg(
        out_svg_path=out_svg,
        width=layout["w"],
        height=layout["h"],
        bg=scheme["bg"],
        margin=layout["margin"],
        logo_h=layout["logo_h"],
        logo_y=layout["logo_y"],
        text_color=scheme["text"],
        highlight_color=scheme["highlight"],
        title_text=conf["title"],
        author_text=conf["author"],
        title_size=layout["title_size"],
        author_size=layout["author_size"],
        title_y=layout["title_y"],
        title_line_height=layout["title_line_height"],
        title_author_gap=layout["title_author_gap"],
    )

    export_png_with_inkscape(out_svg, out_png)

    # For vertical/Shorts: generate an ASS subtitle file (burned in) from proofread SRT + JSON word timing.
    ass_path = None
    if layout["orientation"] == "vertical":
        if srt_path is None:
            raise SystemExit("Vertical output requires a proofread .srt file (same stem as audio).")
        json_path = srt_path.with_suffix(".json")
        if not json_path.exists():
            raise SystemExit(f"Missing JSON next to SRT: {json_path}. Run transcribe.py first.")
        stem = srt_path.with_suffix("")  # /path/audio.srt -> /path/audio
        ass_path = stem.with_suffix(".ass")

        preset = SUB_PRESETS["vertical"]
        # Highlight color defaults to the selected scheme; override here if you want a fixed subtitle accent.
        run_srt_to_ass(
            stem=stem,
            out_ass=ass_path,
            res_w=layout["w"],
            res_h=layout["h"],
            preset=preset,
            highlight_color=scheme["highlight"],
        )


    run_vid_gen(
        out_png,
        audio_path,
        highlight_color=scheme["highlight"],
        orientation=layout["orientation"],
        ass_path=ass_path,
    )
    
    if not KEEP_INTERMEDIATE_SVG:
        out_svg.unlink(missing_ok=True)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[error] Script failed:")
        print(f"{type(e).__name__}: {e}\n")
        traceback.print_exc()
        raise
