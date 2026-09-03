"""Write Nature-style SVG diagrams for Population-DTI-INR proposal PPT."""
from __future__ import annotations

from pathlib import Path

OUT = Path(r"e:\BaiduNetdiskDownload\Population-DTI-INR\ppt_figures\assets")
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#163A5F"
BLUE = "#4C9AFF"
GRAY = "#666666"
LIGHT = "#E8F1FB"
WHITE = "#FFFFFF"
SOFT = "#F7FAFC"


def _box(x, y, w, h, text, *, fill=WHITE, stroke=NAVY, tw=13, bold=False):
    weight = "700" if bold else "500"
    lines = text.split("\n")
    lh = 16
    ty = y + h / 2 - (len(lines) - 1) * lh / 2 + 5
    tspan = "".join(
        f'<tspan x="{x + w/2}" dy="{0 if i == 0 else lh}">{line}</tspan>'
        for i, line in enumerate(lines)
    )
    return f"""
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8"
        fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <text x="{x + w/2}" y="{ty}" text-anchor="middle"
        font-family="Arial, Microsoft YaHei, sans-serif" font-size="{tw}"
        font-weight="{weight}" fill="{NAVY}">{tspan}</text>"""


def _arrow_down(x, y1, y2):
    return f"""
  <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 8}" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="{x-6},{y2-10} {x+6},{y2-10} {x},{y2}" fill="{BLUE}"/>"""


def _arrow_right(x1, y, x2):
    return f"""
  <line x1="{x1}" y1="{y}" x2="{x2 - 8}" y2="{y}" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="{x2-10},{y-6} {x2-10},{y+6} {x2},{y}" fill="{BLUE}"/>"""


def write_title_pipeline():
    # Compact vertical pipeline for title slide
    w, h = 320, 420
    cx = 160
    boxes = [
        (40, "dMRI", LIGHT),
        (120, "INR", WHITE),
        (200, "DTI parameters", WHITE),
        (280, "Microstructure", LIGHT),
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{SOFT}"/>')
    for i, (y, label, fill) in enumerate(boxes):
        parts.append(_box(50, y, 220, 52, label, fill=fill, tw=15, bold=True))
        if i < len(boxes) - 1:
            parts.append(_arrow_down(cx, y + 52, boxes[i + 1][0]))
    parts.append("</svg>")
    (OUT / "title_pipeline.svg").write_text("\n".join(parts), encoding="utf-8")


def write_single_qinr_pipeline():
    w, h = 420, 640
    cx = 210
    steps = [
        "Coordinate (x,y,z)",
        "Fourier Feature",
        "MLP",
        "S0 + D",
        "DTI Forward Model",
        "Signal prediction",
        "MSE(S_pred, S_obs)",
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{WHITE}"/>')
    parts.append(
        f'<text x="{cx}" y="28" text-anchor="middle" font-family="Arial" font-size="16" '
        f'font-weight="700" fill="{NAVY}">Single-QINR Physics Pipeline</text>'
    )
    y0 = 48
    bh = 56
    gap = 22
    for i, label in enumerate(steps):
        y = y0 + i * (bh + gap)
        fill = LIGHT if i in (0, 4, 6) else WHITE
        parts.append(_box(70, y, 280, bh, label, fill=fill, tw=14, bold=(i in (3, 6))))
        if i < len(steps) - 1:
            parts.append(_arrow_down(cx, y + bh, y0 + (i + 1) * (bh + gap)))
    # badge
    parts.append(
        f'<rect x="250" y="580" width="150" height="36" rx="18" fill="{NAVY}"/>'
        f'<text x="325" y="603" text-anchor="middle" font-family="Arial" font-size="11" '
        f'font-weight="700" fill="{WHITE}">Physics self-supervision</text>'
    )
    parts.append("</svg>")
    (OUT / "single_qinr_pipeline.svg").write_text("\n".join(parts), encoding="utf-8")


def write_sparse_sampling():
    w, h = 520, 280
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{WHITE}"/>')
    # Full sampling
    parts.append(
        f'<text x="120" y="36" text-anchor="middle" font-family="Arial" font-size="14" '
        f'font-weight="700" fill="{NAVY}">Full sampling</text>'
    )
    for i in range(5):
        for j in range(4):
            parts.append(
                f'<circle cx="{40 + i * 40}" cy="{70 + j * 36}" r="7" fill="{NAVY}"/>'
            )
    # Sparse
    parts.append(
        f'<text x="390" y="36" text-anchor="middle" font-family="Arial" font-size="14" '
        f'font-weight="700" fill="{NAVY}">Sparse sampling</text>'
    )
    sparse = [(0, 0), (2, 0), (4, 1), (1, 2), (3, 3), (0, 3)]
    for i, j in sparse:
        parts.append(
            f'<circle cx="{310 + i * 40}" cy="{70 + j * 36}" r="7" fill="{BLUE}"/>'
        )
    parts.append(_arrow_right(230, 140, 290))
    parts.append(
        f'<text x="260" y="250" text-anchor="middle" font-family="Arial" font-size="13" '
        f'fill="{GRAY}">Less measurements → higher uncertainty</text>'
    )
    parts.append("</svg>")
    (OUT / "sparse_sampling.svg").write_text("\n".join(parts), encoding="utf-8")


def write_problem_chain():
    w, h = 360, 420
    cx = 180
    steps = ["Sparse sampling", "Signal uncertainty", "Tensor ambiguity", "Parameter instability"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{WHITE}"/>')
    for i, label in enumerate(steps):
        y = 30 + i * 95
        fill = LIGHT if i == 0 else (NAVY if i == 3 else WHITE)
        tc = WHITE if i == 3 else NAVY
        parts.append(
            f'<rect x="40" y="{y}" width="280" height="58" rx="10" fill="{fill}" stroke="{NAVY}" stroke-width="2"/>'
            f'<text x="{cx}" y="{y + 36}" text-anchor="middle" font-family="Arial" font-size="15" '
            f'font-weight="700" fill="{tc}">{label}</text>'
        )
        if i < len(steps) - 1:
            parts.append(_arrow_down(cx, y + 58, 30 + (i + 1) * 95))
    parts.append("</svg>")
    (OUT / "problem_chain.svg").write_text("\n".join(parts), encoding="utf-8")


def write_single_vs_population():
    w, h = 900, 420
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{WHITE}"/>')
    # Left panel
    parts.append(
        f'<rect x="30" y="30" width="340" height="360" rx="14" fill="{SOFT}" stroke="{NAVY}" stroke-width="1.5"/>'
        f'<text x="200" y="70" text-anchor="middle" font-family="Arial" font-size="18" '
        f'font-weight="700" fill="{NAVY}">Single subject</text>'
    )
    for i, lab in enumerate(["Subject", "INR", "DTI"]):
        y = 110 + i * 85
        parts.append(_box(95, y, 210, 55, lab, fill=WHITE, tw=15, bold=True))
        if i < 2:
            parts.append(_arrow_down(200, y + 55, 110 + (i + 1) * 85))
    # Center arrow
    parts.append(_arrow_right(390, 210, 500))
    parts.append(
        f'<text x="445" y="195" text-anchor="middle" font-family="Arial" font-size="12" '
        f'font-weight="700" fill="{BLUE}">Introduce</text>'
        f'<text x="445" y="245" text-anchor="middle" font-family="Arial" font-size="12" '
        f'font-weight="700" fill="{BLUE}">population prior</text>'
    )
    # Right panel
    parts.append(
        f'<rect x="520" y="30" width="350" height="360" rx="14" fill="{LIGHT}" stroke="{NAVY}" stroke-width="1.5"/>'
        f'<text x="695" y="70" text-anchor="middle" font-family="Arial" font-size="18" '
        f'font-weight="700" fill="{NAVY}">Population</text>'
    )
    parts.append(_box(560, 100, 270, 50, "Subjects A / B / C", fill=WHITE, tw=14, bold=True))
    parts.append(_arrow_down(695, 150, 180))
    parts.append(_box(560, 180, 270, 70, "Shared representation\n+ Subject latent z", fill=WHITE, tw=13, bold=True))
    parts.append(_arrow_down(695, 250, 290))
    parts.append(_box(560, 290, 270, 55, "DTI parameters", fill=NAVY, stroke=NAVY, tw=15, bold=True))
    # Override text color for last box - need white text; recreate:
    parts.append("</svg>")
    # Fix last box text - rewrite properly
    svg = "\n".join(parts[:-1])
    # remove incomplete last box by rewriting whole right end
    (OUT / "single_vs_population.svg").write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="{WHITE}"/>
  <rect x="30" y="30" width="340" height="360" rx="14" fill="{SOFT}" stroke="{NAVY}" stroke-width="1.5"/>
  <text x="200" y="70" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="{NAVY}">Single subject</text>
  {_box(95, 110, 210, 55, "Subject", fill=WHITE, tw=15, bold=True)}
  {_arrow_down(200, 165, 195)}
  {_box(95, 195, 210, 55, "INR", fill=WHITE, tw=15, bold=True)}
  {_arrow_down(200, 250, 280)}
  {_box(95, 280, 210, 55, "DTI", fill=WHITE, tw=15, bold=True)}
  {_arrow_right(390, 210, 500)}
  <text x="445" y="190" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="{BLUE}">Introduce</text>
  <text x="445" y="235" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="{BLUE}">population prior</text>
  <rect x="520" y="30" width="350" height="360" rx="14" fill="{LIGHT}" stroke="{NAVY}" stroke-width="1.5"/>
  <text x="695" y="70" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="{NAVY}">Population</text>
  {_box(560, 100, 270, 50, "Subjects A / B / C", fill=WHITE, tw=14, bold=True)}
  {_arrow_down(695, 150, 180)}
  {_box(560, 180, 270, 70, "Shared representation\\n+ Subject latent z", fill=WHITE, tw=13, bold=True)}
  {_arrow_down(695, 250, 290)}
  <rect x="560" y="290" width="270" height="55" rx="8" fill="{NAVY}" stroke="{NAVY}" stroke-width="2"/>
  <text x="695" y="323" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{WHITE}">DTI parameters</text>
</svg>""",
        encoding="utf-8",
    )


def write_population_architecture():
    w, h = 880, 560
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="{WHITE}"/>
  <text x="440" y="36" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="{NAVY}">Population-DTI-INR Architecture</text>

  <!-- Subject branch -->
  <rect x="340" y="60" width="200" height="44" rx="8" fill="{LIGHT}" stroke="{BLUE}" stroke-width="2"/>
  <text x="440" y="88" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="{NAVY}">Subject ID</text>
  <line x1="440" y1="104" x2="440" y2="128" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="434,128 446,128 440,138" fill="{BLUE}"/>
  <rect x="340" y="140" width="200" height="44" rx="8" fill="{BLUE}" stroke="{NAVY}" stroke-width="1.5"/>
  <text x="440" y="168" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="{WHITE}">latent z_s</text>

  <!-- Coordinate branch -->
  <rect x="60" y="200" width="180" height="44" rx="8" fill="{SOFT}" stroke="{NAVY}" stroke-width="2"/>
  <text x="150" y="228" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="{NAVY}">Coordinate x</text>
  <line x1="240" y1="222" x2="300" y2="222" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="300,216 300,228 312,222" fill="{BLUE}"/>

  <!-- Fourier -->
  <rect x="320" y="200" width="240" height="44" rx="8" fill="{WHITE}" stroke="{NAVY}" stroke-width="2"/>
  <text x="440" y="228" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="{NAVY}">Fourier Encoder</text>
  <line x1="440" y1="244" x2="440" y2="268" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="434,268 446,268 440,278" fill="{BLUE}"/>

  <!-- z joins -->
  <line x1="440" y1="184" x2="440" y2="200" stroke="{BLUE}" stroke-width="2" stroke-dasharray="4 3"/>

  <!-- Shared MLP -->
  <rect x="280" y="280" width="320" height="56" rx="10" fill="{NAVY}" stroke="{NAVY}" stroke-width="2"/>
  <text x="440" y="315" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="{WHITE}">Shared MLP  (θ)</text>

  <line x1="440" y1="336" x2="440" y2="360" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="434,360 446,360 440,370" fill="{BLUE}"/>

  <!-- Heads -->
  <rect x="180" y="380" width="180" height="50" rx="8" fill="{LIGHT}" stroke="{NAVY}" stroke-width="2"/>
  <text x="270" y="411" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="{NAVY}">S0 head</text>
  <rect x="520" y="380" width="180" height="50" rx="8" fill="{LIGHT}" stroke="{NAVY}" stroke-width="2"/>
  <text x="610" y="411" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="{NAVY}">D head</text>
  <line x1="360" y1="370" x2="270" y2="380" stroke="{BLUE}" stroke-width="2"/>
  <line x1="520" y1="370" x2="610" y2="380" stroke="{BLUE}" stroke-width="2"/>

  <line x1="270" y1="430" x2="270" y2="455" stroke="{BLUE}" stroke-width="2"/>
  <line x1="610" y1="430" x2="610" y2="455" stroke="{BLUE}" stroke-width="2"/>
  <line x1="270" y1="455" x2="610" y2="455" stroke="{BLUE}" stroke-width="2"/>
  <line x1="440" y1="455" x2="440" y2="470" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="434,470 446,470 440,480" fill="{BLUE}"/>

  <rect x="300" y="482" width="280" height="48" rx="8" fill="{WHITE}" stroke="{NAVY}" stroke-width="2"/>
  <text x="440" y="512" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="{NAVY}">DTI Forward → Predicted signal</text>

  <!-- Legend -->
  <rect x="650" y="140" width="200" height="90" rx="8" fill="{SOFT}" stroke="{GRAY}" stroke-width="1"/>
  <circle cx="675" cy="170" r="8" fill="{NAVY}"/>
  <text x="690" y="175" font-family="Arial" font-size="12" fill="{NAVY}">Shared parameters θ</text>
  <circle cx="675" cy="205" r="8" fill="{BLUE}"/>
  <text x="690" y="210" font-family="Arial" font-size="12" fill="{NAVY}">Subject-specific latent z</text>
</svg>"""
    (OUT / "population_architecture.svg").write_text(svg, encoding="utf-8")


def write_adaptation():
    w, h = 860, 340
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="{WHITE}"/>
  <rect x="40" y="40" width="360" height="260" rx="14" fill="{SOFT}" stroke="{NAVY}" stroke-width="1.5"/>
  <text x="220" y="85" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="{NAVY}">Zero-shot</text>
  {_box(90, 120, 260, 55, "Frozen θ", fill=WHITE, tw=15, bold=True)}
  {_arrow_down(220, 175, 210)}
  {_box(90, 210, 260, 55, "z_new = 0", fill=LIGHT, tw=15, bold=True)}

  <rect x="460" y="40" width="360" height="260" rx="14" fill="{LIGHT}" stroke="{NAVY}" stroke-width="1.5"/>
  <text x="640" y="85" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="{NAVY}">Latent adaptation</text>
  {_box(510, 120, 260, 55, "Frozen θ", fill=WHITE, tw=15, bold=True)}
  {_arrow_down(640, 175, 210)}
  {_box(510, 210, 260, 55, "Optimize z_new", fill=NAVY, stroke=NAVY, tw=15, bold=True)}
  <text x="640" y="243" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{WHITE}">Optimize z_new</text>
</svg>"""
    # Fix duplicate text in last box - rewrite cleanly
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="{WHITE}"/>
  <rect x="40" y="40" width="360" height="260" rx="14" fill="{SOFT}" stroke="{NAVY}" stroke-width="1.5"/>
  <text x="220" y="85" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="{NAVY}">Zero-shot</text>
  <rect x="90" y="120" width="260" height="55" rx="8" fill="{WHITE}" stroke="{NAVY}" stroke-width="2"/>
  <text x="220" y="154" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{NAVY}">Frozen θ</text>
  <line x1="220" y1="175" x2="220" y2="202" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="214,202 226,202 220,212" fill="{BLUE}"/>
  <rect x="90" y="214" width="260" height="55" rx="8" fill="{LIGHT}" stroke="{NAVY}" stroke-width="2"/>
  <text x="220" y="248" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{NAVY}">z_new = 0</text>

  <rect x="460" y="40" width="360" height="260" rx="14" fill="{LIGHT}" stroke="{NAVY}" stroke-width="1.5"/>
  <text x="640" y="85" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="{NAVY}">Latent adaptation</text>
  <rect x="510" y="120" width="260" height="55" rx="8" fill="{WHITE}" stroke="{NAVY}" stroke-width="2"/>
  <text x="640" y="154" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{NAVY}">Frozen θ</text>
  <line x1="640" y1="175" x2="640" y2="202" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="634,202 646,202 640,212" fill="{BLUE}"/>
  <rect x="510" y="214" width="260" height="55" rx="8" fill="{NAVY}" stroke="{NAVY}" stroke-width="2"/>
  <text x="640" y="248" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{WHITE}">Optimize z_new</text>
</svg>"""
    (OUT / "subject_adaptation.svg").write_text(svg, encoding="utf-8")


def write_training():
    w, h = 700, 420
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="{WHITE}"/>
  <rect x="200" y="30" width="300" height="50" rx="8" fill="{LIGHT}" stroke="{NAVY}" stroke-width="2"/>
  <text x="350" y="62" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{NAVY}">Observed DWI</text>
  <line x1="350" y1="80" x2="350" y2="110" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="344,110 356,110 350,120" fill="{BLUE}"/>
  <rect x="180" y="122" width="340" height="55" rx="8" fill="{NAVY}" stroke="{NAVY}" stroke-width="2"/>
  <text x="350" y="156" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{WHITE}">Population-DTI-INR (θ + z_s)</text>
  <line x1="350" y1="177" x2="350" y2="207" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="344,207 356,207 350,217" fill="{BLUE}"/>
  <rect x="200" y="220" width="300" height="50" rx="8" fill="{WHITE}" stroke="{NAVY}" stroke-width="2"/>
  <text x="350" y="252" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{NAVY}">Predicted Signal</text>
  <line x1="350" y1="270" x2="350" y2="300" stroke="{BLUE}" stroke-width="2.5"/>
  <polygon points="344,300 356,300 350,310" fill="{BLUE}"/>
  <rect x="180" y="312" width="340" height="50" rx="8" fill="{LIGHT}" stroke="{NAVY}" stroke-width="2"/>
  <text x="350" y="344" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="{NAVY}">Loss: MSE(S_pred, S_obs)</text>
  <text x="350" y="395" text-anchor="middle" font-family="Arial" font-size="13" fill="{GRAY}">No GT tensor · No FA loss · Physics self-supervision only</text>
</svg>"""
    (OUT / "training_strategy.svg").write_text(svg, encoding="utf-8")


def write_roadmap():
    w, h = 1000, 220
    nodes = [
        (80, "Single-QINR"),
        (280, "Identify\nparameter\ninstability"),
        (480, "Population-\nDTI-INR"),
        (680, "Unseen subject\nvalidation"),
        (880, "DKI\nextension"),
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{WHITE}"/>')
    for i, (x, label) in enumerate(nodes):
        fill = NAVY if i in (0, 2) else LIGHT
        tc = WHITE if i in (0, 2) else NAVY
        lines = label.split("\n")
        parts.append(
            f'<rect x="{x - 70}" y="50" width="140" height="90" rx="12" fill="{fill}" stroke="{NAVY}" stroke-width="2"/>'
        )
        for j, line in enumerate(lines):
            parts.append(
                f'<text x="{x}" y="{95 + (j - (len(lines) - 1) / 2) * 16:.0f}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" font-weight="700" fill="{tc}">{line}</text>'
            )
        if i < len(nodes) - 1:
            x2 = nodes[i + 1][0]
            parts.append(
                f'<line x1="{x + 70}" y1="95" x2="{x2 - 78}" y2="95" stroke="{BLUE}" stroke-width="2.5"/>'
                f'<polygon points="{x2 - 78},{89} {x2 - 78},{101} {x2 - 68},{95}" fill="{BLUE}"/>'
            )
    parts.append("</svg>")
    (OUT / "research_roadmap.svg").write_text("\n".join(parts), encoding="utf-8")


def svg_to_png(svg_path: Path, png_path: Path, scale: float = 2.0):
    """Rasterize SVG via cairosvg if available, else matplotlib fallback via PIL+svglib, else skip."""
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=scale)
        return True
    except Exception:
        pass
    try:
        from io import BytesIO

        from PIL import Image
        from reportlab.graphics import renderPM
        from svglib.svglib import svg2rlg

        drawing = svg2rlg(str(svg_path))
        if drawing is None:
            return False
        png_data = renderPM.drawToString(drawing, fmt="PNG", dpi=144)
        Image.open(BytesIO(png_data)).save(png_path)
        return True
    except Exception:
        return False


def main():
    write_title_pipeline()
    write_single_qinr_pipeline()
    write_sparse_sampling()
    write_problem_chain()
    write_single_vs_population()
    write_population_architecture()
    write_adaptation()
    write_training()
    write_roadmap()

    # Rasterize for PowerPoint embedding reliability
    for svg in OUT.glob("*.svg"):
        png = svg.with_suffix(".png")
        ok = svg_to_png(svg, png)
        print(f"{svg.name} -> {png.name if ok else 'SVG only (no rasterizer)'}")


if __name__ == "__main__":
    main()
