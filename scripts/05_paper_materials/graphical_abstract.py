#!/usr/bin/env python3
"""
Graphical Abstract — Three layout options (A/B/C) for J-BHI submission.

Usage:
    cd /Users/arnold/Documents/Dev/multimodal-depression-detection
    python scripts/05_paper_materials/graphical_abstract.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import warnings; warnings.filterwarnings("ignore")

# ── Palette (same as paper_figures) ──────────────────────────────────
BLU    = "#4E79A7"
RED    = "#F28E2B"
GRY    = "#BAB0AC"
TXT    = "#333333"
TXT_LT = "#888888"
WHITE  = "#FFFFFF"

CORE = Path("results/core")
OUT  = Path("paper/ieee/figures_ieee")
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared data ──────────────────────────────────────────────────────
df = pd.read_csv(CORE / "grand_synthesis.csv").iloc[::-1].reset_index(drop=True)


def ga_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#CCCCCC",
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "axes.grid": False,
    })


def draw_dumbbell(ax, fontscale=1.0):
    """Reusable dumbbell chart on the given axes."""
    y = np.arange(len(df))
    prev_study = None; band_start = 0
    for i, row in df.iterrows():
        if prev_study is not None and row.Study != prev_study:
            if band_start % 2 == 0:
                ax.axhspan(band_start - 0.5, i - 0.5, color="#F8F8F8", zorder=0)
            band_start = i
        prev_study = row.Study
    if band_start % 2 == 0:
        ax.axhspan(band_start - 0.5, len(df) - 0.5, color="#F8F8F8", zorder=0)

    for i, row in df.iterrows():
        p, s = row.R2_personality, row.R2_sensing
        ax.plot([min(p, s), max(p, s)], [i, i], color="#DDDDDD", lw=2, zorder=1,
                solid_capstyle="round")
        ax.scatter(p, i, color=BLU, s=40 * fontscale, zorder=4, edgecolors=WHITE, lw=0.5)
        ax.scatter(s, i, color=RED, s=40 * fontscale, zorder=4, edgecolors=WHITE, lw=0.5)

    labels = [f"{r.Study}  {r.Outcome}" for _, r in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7 * fontscale)
    ax.axvline(0, color=TXT, lw=0.4, alpha=0.4, zorder=1)
    ax.set_xlim(-0.85, 0.7)
    ax.set_xlabel("Cross-Validated R²", fontsize=8 * fontscale)
    ax.tick_params(axis="x", labelsize=7 * fontscale)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.2, zorder=0)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BLU,
               markersize=6, label='Personality'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED,
               markersize=6, label='Sensing'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True,
              edgecolor="#DDDDDD", fancybox=False, fontsize=7 * fontscale,
              borderpad=0.4, handletextpad=0.3)

    n_wins = int(df.Pers_wins.sum())
    ax.text(0.02, 0.04, f"Personality wins {n_wins}/15",
            transform=ax.transAxes, fontsize=7.5 * fontscale, fontweight="bold",
            color=BLU, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=BLU, lw=0.6, alpha=0.95))


# ═══════════════════════════════════════════════════════════════════════
# OPTION A — Key elements re-drawn (clean GA)
# ═══════════════════════════════════════════════════════════════════════
def ga_option_a():
    ga_rcparams()
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.15)

    # ── Left: simplified study summary ───────────────────────────────
    ax_left = fig.add_subplot(gs[0])
    ax_left.set_xlim(0, 10); ax_left.set_ylim(0, 10); ax_left.axis("off")

    # Title
    ax_left.text(5, 9.5, "Can Passive Sensing Replace\nPersonality Questionnaires?",
                 ha="center", va="top", fontsize=14, fontweight="bold", color=TXT)

    # Study summary cards (compact)
    studies = [
        ("Study 1", "StudentLife", "N=28", "10 weeks", "Smartphone"),
        ("Study 2", "NetHealth", "N=722", "4 years", "Fitbit+Phone"),
        ("Study 3", "GLOBEM", "N=809", "1 year", "Fitbit+Phone+GPS"),
    ]
    for i, (name, sub, n, dur, device) in enumerate(studies):
        cy = 7.2 - i * 1.8
        ax_left.add_patch(FancyBboxPatch((0.5, cy - 0.55), 9, 1.3,
            boxstyle="round,pad=0.1", fc="#F7F9FC", ec="#E0E0E0", lw=0.6, zorder=1))
        ax_left.text(1.2, cy + 0.35, name, fontsize=10, fontweight="bold", color=TXT, zorder=2)
        ax_left.text(3.2, cy + 0.35, sub, fontsize=9, color=BLU, fontweight="bold", zorder=2)
        ax_left.text(5.8, cy + 0.35, n, fontsize=9, fontweight="bold", color=TXT, zorder=2)
        ax_left.text(1.2, cy - 0.15, f"{device}  •  {dur}", fontsize=8, color=TXT_LT, zorder=2)

    # Bottom: verdict box
    ax_left.add_patch(FancyBboxPatch((0.5, 1.0), 9, 2.0,
        boxstyle="round,pad=0.15", fc="#FFF8F0", ec=RED, lw=1.0, zorder=1))
    ax_left.text(5, 2.5, "90 days of sensing data  →  R² ≈ 0", ha="center",
                 fontsize=11, fontweight="bold", color=RED, zorder=2)
    ax_left.text(5, 1.5, "A 1-min personality questionnaire  →  R² = 0.09–0.52",
                 ha="center", fontsize=10, fontweight="bold", color=BLU, zorder=2)

    # ── Right: dumbbell ──────────────────────────────────────────────
    ax_right = fig.add_subplot(gs[1])
    draw_dumbbell(ax_right, fontscale=1.0)
    ax_right.set_title("15 Head-to-Head Comparisons", fontsize=10, fontweight="bold", pad=10)

    fig.savefig(OUT / "ga_option_a.png")
    plt.close()
    print("  GA option A saved")


# ═══════════════════════════════════════════════════════════════════════
# OPTION B — Left Fig1 + Right Fig2a (cropped)
# ═══════════════════════════════════════════════════════════════════════
def ga_option_b():
    fig1 = Image.open(OUT / "fig1_overview.png")
    fig2 = Image.open(OUT / "fig2_core.png")

    w1, h1 = fig1.size
    w2, h2 = fig2.size

    # Crop fig2 to top ~58% (dumbbell only, skip incremental bars)
    fig2_top = fig2.crop((0, 0, w2, int(h2 * 0.58)))
    w2t, h2t = fig2_top.size

    # Scale both to same height
    target_h = min(h1, h2t)
    fig1_r = fig1.resize((int(w1 * target_h / h1), target_h), Image.LANCZOS)
    fig2_r = fig2_top.resize((int(w2t * target_h / h2t), target_h), Image.LANCZOS)

    gap = 40
    canvas = Image.new("RGB", (fig1_r.width + fig2_r.width + gap, target_h), (255, 255, 255))
    canvas.paste(fig1_r, (0, 0))
    canvas.paste(fig2_r, (fig1_r.width + gap, 0))

    canvas.save(OUT / "ga_option_b.png", dpi=(300, 300))
    print("  GA option B saved")


# ═══════════════════════════════════════════════════════════════════════
# OPTION C — Top: Fig1c+d (data contrast) + Bottom: Fig2a (dumbbell)
# ═══════════════════════════════════════════════════════════════════════
def ga_option_c():
    ga_rcparams()
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.6], hspace=0.25)

    # ── Top: data contrast (re-drawn from Fig1 c+d) ─────────────────
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.2)

    # Time series
    ax_ts = fig.add_subplot(gs_top[0])
    try:
        steps_data = pd.read_csv("data/raw/globem/INS-W_1/FeatureData/steps.csv",
                                 low_memory=False)
        col = [c for c in steps_data.columns if "avgsumsteps" in c][0]
        pid = steps_data.pid.unique()[2]
        p = steps_data[steps_data.pid == pid].copy()
        p["date"] = pd.to_datetime(p["date"])
        p[col] = pd.to_numeric(p[col], errors="coerce")
        p = p.dropna(subset=[col]).sort_values("date").head(90)
        days = np.arange(len(p))
        vals = p[col].values
    except Exception:
        days = np.arange(90)
        np.random.seed(42)
        vals = np.random.lognormal(8.5, 0.6, 90)

    ax_ts.fill_between(days, 0, vals, color=RED, alpha=0.1, zorder=1)
    ax_ts.plot(days, vals, color=RED, lw=0.6, alpha=0.6, zorder=2)
    ax_ts.set_xlabel("Day of Study", fontsize=9)
    ax_ts.set_ylabel("Daily Steps", fontsize=9)
    ax_ts.set_title("90 Days of Passive Sensing", fontsize=11, fontweight="bold", pad=8)
    ax_ts.text(0.97, 0.93, "All this data  →  R² ≈ 0",
               transform=ax_ts.transAxes, ha="right", va="top", fontsize=10,
               fontweight="bold", color=RED, alpha=0.85,
               bbox=dict(boxstyle="round,pad=0.3", fc="#FFF5F5", ec=RED, lw=0.6, alpha=0.95))

    # Personality card
    ax_card = fig.add_subplot(gs_top[1])
    ax_card.axis("off")
    ax_card.text(-0.08, 0.5, "vs", transform=ax_card.transAxes,
                 ha="center", va="center", fontsize=14, style="italic",
                 color=TXT_LT, fontweight="bold")
    ax_card.add_patch(FancyBboxPatch((0.08, 0.08), 0.85, 0.85,
        boxstyle="round,pad=0.05", fc="#F7F9FC", ec=BLU, lw=1.0,
        transform=ax_card.transAxes))
    ax_card.text(0.5, 0.82, "BFI-10", fontsize=14, fontweight="bold", color=BLU,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.65, "10 items", fontsize=11, color=TXT,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.48, "1 minute", fontsize=16, fontweight="bold", color=BLU,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.32, "Cost: Free", fontsize=9, color=TXT_LT,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.18, "R² = 0.09 – 0.52", fontsize=10, fontweight="bold",
             color=BLU, ha="center", transform=ax_card.transAxes)

    # ── Bottom: dumbbell ─────────────────────────────────────────────
    ax_db = fig.add_subplot(gs[1])
    draw_dumbbell(ax_db, fontscale=1.1)
    ax_db.set_title("Personality vs Sensing: 15 Head-to-Head Comparisons",
                    fontsize=11, fontweight="bold", pad=10)

    fig.savefig(OUT / "ga_option_c.png")
    plt.close()
    print("  GA option C saved")


GRY_LT = "#F0F0F0"

# ═══════════════════════════════════════════════════════════════════════
# GA FINAL — Fig1 re-laid-out in landscape
# ═══════════════════════════════════════════════════════════════════════
def ga_fig1_landscape():
    """Fig1 content in landscape layout for GA.
    Top: pipeline. Bottom-left: 3 study cards. Bottom-right: c+d punchline."""
    ga_rcparams()
    fig = plt.figure(figsize=(13, 6.5))

    gs = fig.add_gridspec(2, 1, height_ratios=[0.4, 2.0], hspace=0.10)

    # ── (a) Pipeline — full width strip ──────────────────────────────
    ax_pipe = fig.add_subplot(gs[0])
    ax_pipe.set_xlim(0, 13); ax_pipe.set_ylim(0, 2.0); ax_pipe.axis("off")

    steps = [
        ("3 Datasets", "N = 1,559", GRY),
        ("Feature\nExtraction", "87 – 2,597", GRY),
        ("ML Models", "Ridge  GBM\nCNN  MOMENT", TXT_LT),
        ("Personality\nvs Sensing", "15 outcomes", BLU),
        ("Robustness", "44 checks", RED),
    ]
    for i, (title, detail, color) in enumerate(steps):
        cx = 1.5 + i * 2.5
        w_box, h_box = 1.9, 1.3
        ax_pipe.add_patch(FancyBboxPatch((cx - w_box/2, 0.3), w_box, h_box,
            boxstyle="round,pad=0.08", fc=color, alpha=0.08,
            ec=color, lw=0.6, zorder=1))
        ax_pipe.text(cx, 1.1, title, ha="center", va="center", fontsize=9,
                fontweight="bold", color=TXT, zorder=2)
        ax_pipe.text(cx, 0.6, detail, ha="center", va="center", fontsize=7,
                color=TXT_LT, zorder=2)
        if i < len(steps) - 1:
            ax_pipe.annotate("", xy=(cx + w_box/2 + 0.22, 0.95),
                        xytext=(cx + w_box/2 + 0.06, 0.95),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=0.7,
                                        mutation_scale=7), zorder=2)

    # ── Bottom: cards (left) + punchline (right) ─────────────────────
    gs_bot = gs[1].subgridspec(1, 2, width_ratios=[1.7, 1], wspace=-0.05)

    # ── (b) Three study cards ────────────────────────────────────────
    ax_cards = fig.add_subplot(gs_bot[0])
    ax_cards.set_xlim(0, 10); ax_cards.set_ylim(0, 7); ax_cards.axis("off")

    studies = [
        {"name": "Study 1", "sub": "StudentLife", "uni": "Dartmouth 2013",
         "n": "N = 28", "pers": "BFI-44", "pt": "5 min",
         "sens": "Smartphone", "sd": "13 modalities", "st": "10 weeks",
         "out": "PHQ-9  PSS  Loneliness  GPA"},
        {"name": "Study 2", "sub": "NetHealth", "uni": "Notre Dame 2015–19",
         "n": "N = 722", "pers": "BFI-44", "pt": "5 min",
         "sens": "Fitbit + Phone", "sd": "28 features", "st": "4 years",
         "out": "CES-D  STAI  BAI  +3 more"},
        {"name": "Study 3", "sub": "GLOBEM", "uni": "U. Washington 2018–21",
         "n": "N = 809", "pers": "BFI-10", "pt": "1 min",
         "sens": "Fitbit + Phone + GPS", "sd": "19–2,597 features", "st": "1 year",
         "out": "BDI-II  STAI  PSS  +2 more"},
    ]

    for i, s in enumerate(studies):
        cx = 1.7 + i * 2.8
        ax_cards.add_patch(FancyBboxPatch((cx - 1.22, 0.18), 2.44, 6.4,
            boxstyle="round,pad=0.1", fc="#F5F5F5", ec="none", zorder=0))
        ax_cards.add_patch(FancyBboxPatch((cx - 1.25, 0.2), 2.44, 6.4,
            boxstyle="round,pad=0.1", fc=WHITE, ec="#E0E0E0", lw=0.5, zorder=1))

        y0 = 6.2
        ax_cards.text(cx, y0, s["name"], ha="center", fontsize=9,
                fontweight="bold", color=TXT, zorder=2)
        ax_cards.text(cx, y0 - 0.3, s["sub"], ha="center", fontsize=7.5,
                color=BLU, fontweight="bold", zorder=2)
        ax_cards.text(cx, y0 - 0.55, s["uni"], ha="center", fontsize=6,
                color=TXT_LT, zorder=2)

        ax_cards.add_patch(FancyBboxPatch((cx - 0.45, y0 - 1.0), 0.9, 0.3,
            boxstyle="round,pad=0.04", fc=BLU, ec="none", zorder=2))
        ax_cards.text(cx, y0 - 0.85, s["n"], ha="center", va="center", fontsize=7,
                fontweight="bold", color=WHITE, zorder=3)

        qy = 4.0
        ax_cards.add_patch(FancyBboxPatch((cx - 0.95, qy), 1.9, 0.75,
            boxstyle="round,pad=0.05", fc=BLU, alpha=0.05, ec=BLU, lw=0.4, zorder=1))
        ax_cards.text(cx - 0.8, qy + 0.6, "QUESTIONNAIRE", fontsize=4,
                fontweight="bold", color=BLU, alpha=0.6, zorder=2)
        ax_cards.text(cx, qy + 0.38, s["pers"], ha="center", fontsize=7, color=TXT, zorder=2)
        ax_cards.text(cx, qy + 0.1, s["pt"], ha="center", fontsize=10,
                fontweight="bold", color=BLU, zorder=2)

        ax_cards.text(cx, 3.75, "vs", ha="center", fontsize=6, color=TXT_LT,
                style="italic", zorder=2)

        sy = 1.6
        ax_cards.add_patch(FancyBboxPatch((cx - 0.95, sy), 1.9, 1.9,
            boxstyle="round,pad=0.05", fc=RED, alpha=0.04, ec=RED, lw=0.4, zorder=1))
        ax_cards.text(cx - 0.8, sy + 1.72, "PASSIVE SENSING", fontsize=4,
                fontweight="bold", color=RED, alpha=0.6, zorder=2)
        ax_cards.text(cx, sy + 1.35, s["sens"], ha="center", fontsize=7, color=TXT, zorder=2)
        ax_cards.text(cx, sy + 1.08, s["sd"], ha="center", fontsize=6, color=TXT_LT, zorder=2)
        ax_cards.text(cx, sy + 0.25, s["st"], ha="center", fontsize=11,
                fontweight="bold", color=RED, zorder=2)

        ax_cards.annotate("", xy=(cx, 1.25), xytext=(cx, 1.5),
                    arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=0.6,
                                    mutation_scale=6), zorder=2)
        ax_cards.text(cx, 0.95, s["out"], ha="center", fontsize=5.5, color=TXT_LT,
                zorder=2, style="italic")

    ax_cards.add_patch(FancyBboxPatch((0.35, 0.05), 8.25, 0.45,
        boxstyle="round,pad=0.06", fc=GRY_LT, ec="none", zorder=1))
    ax_cards.text(4.3, 0.27, "15 head-to-head comparisons  |  44 robustness analyses  |  4 ML architectures",
            ha="center", fontsize=6, color=TXT, zorder=2)

    # ── (c+d) Punchline — right side, stacked ────────────────────────
    gs_right = gs_bot[1].subgridspec(3, 1, height_ratios=[0.75, 0.12, 0.5], hspace=0.12)

    # Time series
    ax_ts = fig.add_subplot(gs_right[0])
    try:
        steps_data = pd.read_csv("data/raw/globem/INS-W_1/FeatureData/steps.csv",
                                  low_memory=False)
        col = [c for c in steps_data.columns if "avgsumsteps" in c][0]
        pid = steps_data.pid.unique()[2]
        p = steps_data[steps_data.pid == pid].copy()
        p["date"] = pd.to_datetime(p["date"])
        p[col] = pd.to_numeric(p[col], errors="coerce")
        p = p.dropna(subset=[col]).sort_values("date").head(90)
        days = np.arange(len(p))
        vals = p[col].values
    except Exception:
        days = np.arange(90)
        np.random.seed(42)
        vals = np.random.lognormal(8.5, 0.6, 90)

    ax_ts.fill_between(days, 0, vals, color=RED, alpha=0.1, zorder=1)
    ax_ts.plot(days, vals, color=RED, lw=0.5, alpha=0.6, zorder=2)
    ax_ts.set_xlabel("Day of Study", fontsize=6)
    ax_ts.set_ylabel("Daily Steps", fontsize=6)
    ax_ts.set_title("90 Days of Passive Sensing", fontsize=7, fontweight="bold", pad=4)
    ax_ts.tick_params(labelsize=5.5)
    ax_ts.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax_ts.text(0.93, 0.86, r"All this data  $\rightarrow$  R$^2$ $\approx$ 0",
            transform=ax_ts.transAxes, ha="right", va="top", fontsize=7.5,
            fontweight="bold", color=RED, alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF5F5", ec=RED,
                      lw=0.5, alpha=0.95))

    # "vs" spacer
    ax_vs = fig.add_subplot(gs_right[1])
    ax_vs.axis("off")
    ax_vs.text(0.5, 0.1, "vs", transform=ax_vs.transAxes,
               ha="center", va="center", fontsize=11, style="italic",
               color=TXT, fontweight="bold")

    # BFI card
    ax_card = fig.add_subplot(gs_right[2])
    ax_card.axis("off")

    ax_card.add_patch(FancyBboxPatch((0.1, 0.05), 0.8, 0.9,
        boxstyle="round,pad=0.05", fc="#F7F9FC", ec=BLU, lw=0.6,
        transform=ax_card.transAxes))
    ax_card.text(0.5, 0.82, "BFI-10", fontsize=10, fontweight="bold", color=BLU,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.66, "10 items", fontsize=7.5, color=TXT,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.49, "1 minute", fontsize=11, fontweight="bold", color=BLU,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.33, "Cost: Free", fontsize=6, color=TXT_LT,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.17, r"R$^2$ = 0.09 – 0.52", fontsize=7, fontweight="bold",
             color=BLU, ha="center", transform=ax_card.transAxes)

    fig.savefig(OUT / "ga_final.png")
    plt.close()
    print("  GA final (landscape Fig1) saved")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating graphical abstract...")
    ga_fig1_landscape()
    print(f"Saved to {OUT}/ga_final.png")
