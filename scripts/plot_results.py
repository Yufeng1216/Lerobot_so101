"""Draw the per-spot success chart for the README from the evaluation sheets in results/.

Reads every results/<policy>.md (the table rows `| # | spot | result | ... |` filled in after a
real-robot evaluation, see docs/08_evaluation.md) and writes assets/results.svg: one group of
bars per start spot, one bar per policy, plus each policy's overall success rate in the legend.
SVG is plain text, renders on GitHub, and needs no plotting library.

Usage:
    python scripts/plot_results.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPOTS = "①②③④⑤"
ORDER = ["act", "diffusion", "smolvla", "pi05", "vlm_agent"]  # README order; missing ones are skipped
NAMES = {"act": "ACT", "diffusion": "Diffusion Policy", "smolvla": "SmolVLA", "pi05": "π₀.₅", "vlm_agent": "VLM agent"}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
ROW = re.compile(r"^\|\s*\d+\s*\|\s*([①②③④⑤])\s*\|\s*(✅|🟡|❌)\s*\|")


def read_sheet(path: Path) -> dict[str, list[bool]]:
    """{spot: [success?, ...]} from one evaluation sheet."""
    per_spot: dict[str, list[bool]] = {s: [] for s in SPOTS}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if m:
            per_spot[m.group(1)].append(m.group(2) == "✅")
    return per_spot


def trials_label(sheets) -> str:
    """'8 trials each', or a range if policies were evaluated a different number of times."""
    counts = sorted({len(t) for _, sheet in sheets for t in sheet.values()})
    return f"{counts[0]} trials each" if len(counts) == 1 else f"{counts[0]}-{counts[-1]} trials each"


def main() -> None:
    sheets = [(p, read_sheet(ROOT / "results" / f"{p}.md")) for p in ORDER if (ROOT / "results" / f"{p}.md").exists()]
    W, H, left, top, bottom = 760, 310, 50, 62, 40
    plot_h = H - top - bottom
    group_w = (W - left - 20) / len(SPOTS)
    bar_w = min(28, (group_w - 16) / max(len(sheets), 1))

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{left}" y="18" font-size="14" font-weight="bold">Real-robot success rate per start spot ({trials_label(sheets)})</text>']
    for pct in (0, 25, 50, 75, 100):  # horizontal grid + y labels
        y = top + plot_h * (1 - pct / 100)
        svg.append(f'<line x1="{left}" x2="{W - 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#666">{pct}%</text>')
    for g, spot in enumerate(SPOTS):
        gx = left + g * group_w
        span = bar_w * len(sheets)
        for k, (_, sheet) in enumerate(sheets):
            trials = sheet[spot]
            rate = sum(trials) / len(trials) if trials else 0
            x = gx + (group_w - span) / 2 + k * bar_w
            h = plot_h * rate
            svg.append(f'<rect x="{x:.1f}" y="{top + plot_h - h:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}" fill="{COLORS[k]}"/>')
            svg.append(f'<text x="{x + bar_w / 2 - 1:.1f}" y="{top + plot_h - h - 3:.1f}" text-anchor="middle" font-size="10" fill="#333">{sum(trials)}/{len(trials)}</text>')
        svg.append(f'<text x="{gx + group_w / 2:.1f}" y="{H - bottom + 18}" text-anchor="middle" font-size="14">{spot}</text>')
    x, y = left, 40
    for k, (name, sheet) in enumerate(sheets):  # legend row under the title: name + overall rate
        allt = [t for v in sheet.values() for t in v]
        label = f"{NAMES[name]} ({sum(allt)}/{len(allt)})"
        svg.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" fill="{COLORS[k]}"/>')
        svg.append(f'<text x="{x + 14}" y="{y}">{label}</text>')
        x += 14 + 6.8 * len(label) + 18          # ~6.8 px per character at font-size 12
    svg.append("</svg>")
    out = ROOT / "assets" / "results.svg"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(svg), encoding="utf-8")
    print(f"saved {out}  policies: {[n for n, _ in sheets]}")


if __name__ == "__main__":
    main()
