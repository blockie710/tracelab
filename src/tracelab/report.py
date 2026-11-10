"""Static HTML report generator — no JS, no CDN, one self-contained file."""

from __future__ import annotations

import html
import json
import math
from typing import Dict, List, Sequence

W, H = 460, 300
PAD = 44


def _esc(s) -> str:
    return html.escape(str(s))


def _roc_svg(points: Sequence[Sequence[float]], auc: float) -> str:
    def x(f): return PAD + f * (W - 2 * PAD)
    def y(t): return H - PAD - t * (H - 2 * PAD)

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(f):.1f},{y(t):.1f}" for i, (f, t) in enumerate(points)
    )
    ticks = []
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        ticks.append(
            f'<line x1="{x(v):.1f}" y1="{y(0):.1f}" x2="{x(v):.1f}" y2="{y(1):.1f}" class="grid"/>'
            f'<text x="{x(v):.1f}" y="{H - PAD + 18}" class="tick" text-anchor="middle">{v:g}</text>'
            f'<line x1="{x(0):.1f}" y1="{y(v):.1f}" x2="{x(1):.1f}" y2="{y(v):.1f}" class="grid"/>'
            f'<text x="{PAD - 10}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>'
        )
    return f"""
<svg viewBox="0 0 {W} {H}" role="img" aria-label="ROC curve, AUC {auc:.3f}">
  {''.join(ticks)}
  <line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" class="diag"/>
  <path d="{path}" class="roc"/>
  <text x="{W/2}" y="{H-6}" class="axis" text-anchor="middle">false positive rate</text>
  <text x="14" y="{H/2}" class="axis" transform="rotate(-90 14 {H/2})" text-anchor="middle">true positive rate</text>
</svg>"""


def _hist_svg(dists: Dict[str, List[float]], bins: int = 16) -> str:
    """Grouped histogram: one bar per class per bin, so nothing is hidden."""
    colours = {"human": "#6b7280", "snap": "#ef4444", "smoothed": "#f59e0b"}
    counts: Dict[str, List[int]] = {}
    for kind, vals in dists.items():
        c = [0] * bins
        for v in vals:
            c[min(bins - 1, max(0, int(v * bins)))] += 1
        counts[kind] = c

    HH = 340                      # this chart is taller: it needs an axis title + legend
    pad_l, pad_r, pad_t = 56, 18, 16
    plot_w = W - pad_l - pad_r
    plot_h = 210
    base_y = pad_t + plot_h

    top = max(max(c) for c in counts.values()) or 1
    bin_w = plot_w / bins
    bar_w = bin_w / (len(counts) + 0.8)

    bars = []
    for k, kind in enumerate(counts):
        for i, v in enumerate(counts[kind]):
            h = v / top * plot_h
            bars.append(
                f'<rect x="{pad_l + i*bin_w + k*bar_w + 1.0:.1f}" y="{base_y - h:.1f}" '
                f'width="{max(1.0, bar_w - 1.6):.1f}" height="{h:.1f}" '
                f'fill="{colours[kind]}" opacity="0.92"><title>{kind}: {v}</title></rect>'
            )

    grid = []
    for frac in (0.0, 0.5, 1.0):
        gy = base_y - frac * plot_h
        grid.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w:.1f}" y2="{gy:.1f}" class="grid"/>'
            f'<text x="{pad_l - 9}" y="{gy + 4:.1f}" class="tick" text-anchor="end">'
            f'{int(round(frac * top))}</text>'
        )

    xt = []
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        xt.append(
            f'<text x="{pad_l + v * plot_w:.1f}" y="{base_y + 22:.1f}" class="tick" '
            f'text-anchor="middle">{v:g}</text>'
        )

    legend = "".join(
        f'<rect x="{pad_l + i * 140}" y="{HH - 26}" width="9" height="9" fill="{colours[k]}"/>'
        f'<text x="{pad_l + i * 140 + 15}" y="{HH - 18}" class="tick">{k}</text>'
        for i, k in enumerate(counts)
    )

    return f"""
<svg viewBox="0 0 {W} {HH}" role="img" aria-label="detector score distribution by class">
  {''.join(grid)}
  {''.join(bars)}
  {''.join(xt)}
  <text x="{pad_l + plot_w/2:.0f}" y="{base_y + 44:.0f}" class="axis" text-anchor="middle">detector score</text>
  <text x="16" y="{pad_t + plot_h/2:.0f}" class="axis" transform="rotate(-90 16 {pad_t + plot_h/2:.0f})" text-anchor="middle">traces per bin</text>
  {legend}
</svg>"""


def _bar(v: float, vmax: float, colour: str) -> str:
    pct = 0 if vmax <= 0 else min(100, abs(v) / vmax * 100)
    return (
        f'<span class="bar"><span class="fill" style="width:{pct:.1f}%;'
        f'background:{colour}"></span></span>'
    )


def render(result: Dict) -> str:
    m = result["metrics"]
    cfg = result["config"]
    fmeans = result["feature_means"]
    weights = result["weights"]

    feat_rows = []
    for name in cfg["features"]:
        row = fmeans[name]
        vmax = max(abs(row[k]) for k in row) or 1.0
        w = weights[name]
        sign = "+" if w >= 0 else "−"
        colour = "#ef4444" if w >= 0 else "#22d3ee"
        feat_rows.append(
            f"<tr><td class='mono'>{_esc(name)}</td>"
            f"<td>{_bar(row['human'], vmax, '#6b7280')} {row['human']:.2f}</td>"
            f"<td>{_bar(row['snap'], vmax, '#ef4444')} {row['snap']:.2f}</td>"
            f"<td>{_bar(row['smoothed'], vmax, '#f59e0b')} {row['smoothed']:.2f}</td>"
            f"<td class='mono' style='color:{colour}'>{sign}{abs(w):.2f}</td></tr>"
        )

    prev_rows = "".join(
        f"<tr><td>{p['prevalence']*100:.1f}%</td>"
        f"<td>{p['precision']*100:.1f}%</td>"
        f"<td>{p['flagged_per_10k']:.0f}</td></tr>"
        for p in result["prevalence_table"]
    )

    def evidence_card(card: Dict, title: str, tone: str) -> str:
        contribs = "".join(
            f"<li><span class='mono'>{_esc(n)}</span>"
            f"<span class='contrib'>{v:+.2f}</span></li>"
            for n, v in card["contributions"]
        )
        return f"""
<div class="card evidence {tone}">
  <div class="small">{_esc(title)}</div>
  <div class="score">{card['score']:.3f}</div>
  <div class="small">reference label: <span class="mono">{_esc(card['kind'])}</span></div>
  <div class="small" style="margin-top:10px">largest logit contributions</div>
  <ul class="contribs">{contribs}</ul>
</div>"""

    lo, hi = m["auc_ci95"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tracelab — kinematic trace analysis report</title>
<style>
  :root {{
    --bg:#0b0d10; --panel:#12161b; --line:#232a33; --fg:#e6edf3; --dim:#8b98a5; --accent:#22d3ee;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:40px 22px 80px; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-0.02em; }}
  h2 {{ font-size:15px; text-transform:uppercase; letter-spacing:0.09em; color:var(--dim);
    margin:38px 0 12px; font-weight:600; }}
  .sub {{ color:var(--dim); margin:0 0 26px; }}
  .grid {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); }}
  .kpis {{ grid-template-columns:repeat(4,1fr); }}
  @media (max-width:820px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .kpi {{ font-size:30px; font-weight:650; letter-spacing:-0.02em; font-variant-numeric:tabular-nums; }}
  .small {{ color:var(--dim); font-size:12px; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th,td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); font-size:13px; }}
  th {{ color:var(--dim); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:0.07em; }}
  .bar {{ display:inline-block; width:70px; height:6px; background:#1c232b; border-radius:3px;
    margin-right:6px; vertical-align:middle; overflow:hidden; }}
  .fill {{ display:block; height:100%; }}
  svg {{ width:100%; height:auto; }}
  .grid line {{ stroke:#1f262e; stroke-width:1; }}
  .diag {{ stroke:#39424d; stroke-dasharray:4 4; }}
  .roc {{ fill:none; stroke:var(--accent); stroke-width:2.2; }}
  .tick {{ fill:#7b8794; font-size:11px; font-family:ui-monospace,monospace; }}
  .axis {{ fill:#7b8794; font-size:11px; }}
  .evidence .score {{ font-size:34px; font-weight:700; font-variant-numeric:tabular-nums; }}
  .evidence.bad {{ border-color:#5c1f1f; }}
  .evidence.ok {{ border-color:#1f4a5c; }}
  .contribs {{ list-style:none; padding:0; margin:6px 0 0; }}
  .contribs li {{ display:flex; justify-content:space-between; padding:3px 0;
    border-bottom:1px dashed #1c232b; }}
  .contrib {{ font-variant-numeric:tabular-nums; color:var(--dim); }}
  .note {{ color:var(--dim); font-size:12.5px; border-left:2px solid var(--line);
    padding-left:12px; margin:14px 0; }}
  code {{ background:#171d24; padding:2px 6px; border-radius:5px; font-size:12.5px; }}
</style></head><body><div class="wrap">

<h1>tracelab</h1>
<p class="sub">Kinematic analysis of high-rate input telemetry. Synthetic benchmark,
seeded and reproducible — no client agent, no kernel driver, no model weights.</p>

<div class="grid kpis">
  <div class="card"><div class="small">AUC (bot vs human)</div>
    <div class="kpi">{m['auc']:.3f}</div>
    <div class="small">95% CI {lo:.3f}–{hi:.3f}</div></div>
  <div class="card"><div class="small">Recall at 1% FPR</div>
    <div class="kpi">{m['tpr_at_1pct_fpr']*100:.1f}%</div>
    <div class="small">threshold {m['threshold']:.3f}</div></div>
  <div class="card"><div class="small">Realised FPR</div>
    <div class="kpi">{m['realised_fpr']*100:.2f}%</div>
    <div class="small">{m['fp']} false positives / {m['tn']+m['fp']} humans</div></div>
  <div class="card"><div class="small">Hard case: smoothed actor</div>
    <div class="kpi">{result['per_kind']['smoothed']['recall_at_thr']*100:.1f}%</div>
    <div class="small">AUC {result['per_kind']['smoothed']['auc']:.3f}</div></div>
</div>

<h2>ROC</h2>
<div class="card">{_roc_svg(result['roi_of_roc'], m['auc'])}</div>

<h2>Score distribution</h2>
<div class="card">{_hist_svg(result['score_distributions'])}</div>
<p class="note">Note the overlap in the middle: raw snaps separate almost perfectly, smoothed
actors do not. That gap is the honest perimeter of this method.</p>

<h2>Features</h2>
<div class="card"><table>
  <thead><tr><th>feature</th><th>human</th><th>snap</th><th>smoothed</th><th>weight</th></tr></thead>
  <tbody>{''.join(feat_rows)}</tbody>
</table></div>

<h2>Precision at realistic positive prevalence</h2>
<div class="card"><table>
  <thead><tr><th>positive prevalence</th><th>precision of a ban</th><th>flagged per 10k players</th></tr></thead>
  <tbody>{prev_rows}</tbody>
</table></div>
<p class="note">This is the table that matters. At a 1% false-positive rate and a 2% positive
rate, roughly two in five flagged cases would be wrong. Any headline number that
quotes only accuracy has skipped this arithmetic.</p>

<h2>Evidence card</h2>
<div class="grid">
{evidence_card(result['evidence']['flagged'], 'lowest-scoring positive in test set', 'bad')}
{evidence_card(result['evidence']['closest_human'], 'highest-scoring negative in test set', 'ok')}
</div>
<p class="note">A verdict ships with its reasons. That is the difference between a decision
that survives review and one that becomes a dispute.</p>

<h2>Reproduce</h2>
<div class="card">
<div class="small">seed {cfg['seed']} · {cfg['n_train_per_kind']} train / {cfg['n_test_per_kind']} test traces per class · {cfg['tick_hz']} Hz · L2 {cfg['l2']}</div>
<pre class="mono" style="margin:12px 0 0; overflow:auto">pip install -e .
tracelab demo --out demo/index.html --json demo/results.json</pre>
</div>

</div></body></html>"""


def write_report(result: Dict, path: str) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(result))
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
