#!/usr/bin/env python3
"""
IEEE Double-Column Figures — Wrapper around paper_figures.py
=============================================================
Generates all 11 figures (5 main + 6 supplementary) with IEEE-optimized
dimensions and font sizes.

IEEE column widths:
  - Single column: 3.5 in (88 mm)
  - Full width (figure*): 7.0 in (178 mm)

Usage:
    cd /Users/arnold/Documents/Dev/multimodal-depression-detection
    python scripts/05_paper_materials/paper_figures_ieee.py
    python scripts/05_paper_materials/paper_figures_ieee.py fig1 fig3
"""

import sys, os, numpy as np, pandas as pd
from pathlib import Path
from unittest.mock import patch
from functools import wraps
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.interpolate import make_interp_spline

# ── Import original module ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import paper_figures as pf

# ── Paths ────────────────────────────────────────────────────────────
CORE = pf.CORE
ROB  = pf.ROB
BY   = pf.BY
OUT  = Path("paper/ieee/figures_ieee")
OUT.mkdir(parents=True, exist_ok=True)

# ── Reuse palette ────────────────────────────────────────────────────
BLU    = pf.BLU
RED    = pf.RED
GRY    = pf.GRY
GRY_LT = pf.GRY_LT
TXT    = pf.TXT
TXT_LT = pf.TXT_LT
WHITE  = pf.WHITE
BG     = pf.BG


def setup_ieee():
    """IEEE-optimized rcParams: smaller fonts for double-column layout."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7,
        "axes.labelweight": "medium",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.4,
        "axes.edgecolor": "#CCCCCC",
        "axes.facecolor": WHITE,
        "axes.titlepad": 8,
        "axes.labelpad": 4,
        "figure.facecolor": WHITE,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.grid": False,
        "grid.color": "#EEEEEE",
        "grid.linewidth": 0.2,
        "grid.alpha": 1.0,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 0.9,
        "patch.linewidth": 0.4,
    })


def plabel(ax, s, x=-0.10, y=1.08):
    """Panel label — smaller for IEEE."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=TXT, va="top", ha="left")


def clean_axis(ax, grid_axis=None):
    if grid_axis:
        ax.grid(axis=grid_axis, color="#EEEEEE", linewidth=0.2, zorder=0)
    ax.tick_params(axis="both", which="both", length=2, width=0.4, color="#CCCCCC")


def badge(ax, text, x, y, color=BLU, fontsize=6.5, transform=None):
    if transform is None:
        transform = ax.transAxes
    ax.text(x, y, text, transform=transform, fontsize=fontsize,
            fontweight="bold", color=color, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=color,
                      lw=0.6, alpha=0.95))


# ═══════════════════════════════════════════════════════════════════════
# FIG 1 — FULL WIDTH (7.0 x 4.5) — Study Overview (compact)
# ═══════════════════════════════════════════════════════════════════════
def fig1_ieee():
    """Fig 1: study cards only. Full width, compact."""
    fig = plt.figure(figsize=(7.0, 4.5))

    # ── Three study cards ────────────────────────────────────────────
    ax_cards = fig.add_subplot(111)
    ax_cards.set_xlim(-0.1, 9.1); ax_cards.set_ylim(0, 7); ax_cards.axis("off")

    studies = [
        {"name": "Study 1", "sub": "StudentLife", "uni": "Dartmouth 2013",
         "n": "N = 28", "pers": "BFI-44", "pt": "5 min",
         "sens": "Smartphone", "sd": "13 modalities", "st": "10 weeks",
         "out": "PHQ-9  PSS  Loneliness  GPA"},
        {"name": "Study 2", "sub": "NetHealth", "uni": "Notre Dame 2015-19",
         "n": "N = 722", "pers": "BFI-44", "pt": "5 min",
         "sens": "Fitbit + Phone", "sd": "28 features", "st": "4 years",
         "out": "CES-D  STAI  BAI  +3 more"},
        {"name": "Study 3", "sub": "GLOBEM", "uni": "U. Washington 2018-21",
         "n": "N = 809", "pers": "BFI-10", "pt": "1 min",
         "sens": "Fitbit + Phone + GPS", "sd": "19-2,597 features", "st": "1 year",
         "out": "BDI-II  STAI  PSS  +2 more"},
    ]

    for i, s in enumerate(studies):
        cx = 1.7 + i * 2.8
        ax_cards.add_patch(FancyBboxPatch((cx - 1.22, 0.18), 2.44, 6.4,
            boxstyle="round,pad=0.1", fc="#F5F5F5", ec="none", zorder=0))
        ax_cards.add_patch(FancyBboxPatch((cx - 1.25, 0.2), 2.44, 6.4,
            boxstyle="round,pad=0.1", fc=WHITE, ec="#E0E0E0", lw=0.5, zorder=1))

        y0 = 6.2
        ax_cards.text(cx, y0, s["name"], ha="center", fontsize=8,
                fontweight="bold", color=TXT, zorder=2)
        ax_cards.text(cx, y0 - 0.28, s["sub"], ha="center", fontsize=6.5,
                color=BLU, fontweight="bold", zorder=2)
        ax_cards.text(cx, y0 - 0.5, s["uni"], ha="center", fontsize=5.5,
                color=TXT_LT, zorder=2)

        ax_cards.add_patch(FancyBboxPatch((cx - 0.4, y0 - 0.95), 0.8, 0.28,
            boxstyle="round,pad=0.04", fc=BLU, ec="none", zorder=2))
        ax_cards.text(cx, y0 - 0.81, s["n"], ha="center", va="center", fontsize=6,
                fontweight="bold", color=WHITE, zorder=3)

        qy = 4.0
        ax_cards.add_patch(FancyBboxPatch((cx - 0.95, qy), 1.9, 0.75,
            boxstyle="round,pad=0.05", fc=BLU, alpha=0.05, ec=BLU, lw=0.4, zorder=1))
        ax_cards.text(cx - 0.8, qy + 0.6, "QUESTIONNAIRE", fontsize=3.5,
                fontweight="bold", color=BLU, alpha=0.6, zorder=2)
        ax_cards.text(cx, qy + 0.38, s["pers"], ha="center", fontsize=6, color=TXT, zorder=2)
        ax_cards.text(cx, qy + 0.1, s["pt"], ha="center", fontsize=9,
                fontweight="bold", color=BLU, zorder=2)

        ax_cards.text(cx, 3.75, "vs", ha="center", fontsize=5.5, color=TXT_LT,
                style="italic", zorder=2)

        sy = 1.6
        ax_cards.add_patch(FancyBboxPatch((cx - 0.95, sy), 1.9, 1.9,
            boxstyle="round,pad=0.05", fc=RED, alpha=0.04, ec=RED, lw=0.4, zorder=1))
        ax_cards.text(cx - 0.8, sy + 1.72, "PASSIVE SENSING", fontsize=3.5,
                fontweight="bold", color=RED, alpha=0.6, zorder=2)
        ax_cards.text(cx, sy + 1.35, s["sens"], ha="center", fontsize=6, color=TXT, zorder=2)
        ax_cards.text(cx, sy + 1.08, s["sd"], ha="center", fontsize=5.5, color=TXT_LT, zorder=2)
        ax_cards.text(cx, sy + 0.25, s["st"], ha="center", fontsize=10,
                fontweight="bold", color=RED, zorder=2)

        ax_cards.annotate("", xy=(cx, 1.25), xytext=(cx, 1.5),
                    arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=0.6,
                                    mutation_scale=6), zorder=2)
        ax_cards.text(cx, 0.95, s["out"], ha="center", fontsize=5, color=TXT_LT,
                zorder=2, style="italic")

    ax_cards.add_patch(FancyBboxPatch((0.35, 0.05), 8.25, 0.45,
        boxstyle="round,pad=0.06", fc=GRY_LT, ec="none", zorder=1))
    ax_cards.text(4.3, 0.27, "15 head-to-head comparisons  |  44 robustness analyses  |  4 ML architectures",
            ha="center", fontsize=5.5, color=TXT, zorder=2)

    fig.savefig(OUT / "fig1_overview.png")
    plt.close(); print("  fig1 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 2 — SINGLE COLUMN (3.5 x 6) — Core Results
# ═══════════════════════════════════════════════════════════════════════
def fig2_ieee():
    """Fig 2: dumbbell + incremental bars. Single column."""
    fig = plt.figure(figsize=(3.5, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.4, 0.8], hspace=0.38)

    # ── (a) Dumbbell ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a", x=-0.14, y=1.06)
    df = pd.read_csv(CORE / "grand_synthesis.csv")
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))

    prev_study = None
    band_start = 0
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
        ax.plot([min(p, s), max(p, s)], [i, i], color="#DDDDDD", lw=1.5, zorder=1,
                solid_capstyle="round")
        ax.scatter(p, i, color=BLU, s=25, zorder=4, edgecolors=WHITE, lw=0.5)
        ax.scatter(s, i, color=RED, s=25, zorder=4, edgecolors=WHITE, lw=0.5)

    labels = [f"{r.Study}  {r.Outcome}" for _, r in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5)

    ax.axvline(0, color=TXT, lw=0.4, alpha=0.4, zorder=1)
    ax.set_xlim(-0.8, 0.7)
    ax.set_xlabel("Cross-Validated R²", fontsize=6)
    ax.set_title("Personality vs Sensing:\n15 Head-to-Head Comparisons",
                 fontsize=7, pad=8)
    clean_axis(ax, grid_axis="x")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BLU,
               markersize=5, label='Personality'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED,
               markersize=5, label='Sensing'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True,
              edgecolor="#DDDDDD", fancybox=False, fontsize=5.5,
              borderpad=0.4, handletextpad=0.3)

    n_wins = int(df.Pers_wins.sum())
    badge(ax, f"11/11 confirmatory ({n_wins}/15 overall)", 0.02, 0.04, color=BLU, fontsize=5.5)

    # ── (b) Incremental validity ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1]); plabel(ax3, "b", x=-0.14, y=1.08)
    inc = pd.read_csv(CORE / "incremental_validity.csv")
    labels_inc = [f"{r.Study} {r.Outcome}" for _, r in inc.iterrows()]
    y_inc = np.arange(len(inc))
    colors_inc = [BLU if not r.sig_fdr else "#8B5CF6" for _, r in inc.iterrows()]
    ax3.barh(y_inc, inc.Delta_R2, color=colors_inc, edgecolor=WHITE, lw=0.4,
             height=0.6, zorder=3, alpha=0.8)
    for i, row in inc.iterrows():
        sig = "*" if row.sig_fdr else ""
        ax3.text(row.Delta_R2 + 0.001, i, f"{row.Delta_R2:.3f}{sig}",
                 va="center", fontsize=4.5, color=TXT)
    ax3.set_yticks(y_inc)
    ax3.set_yticklabels(labels_inc, fontsize=5)
    ax3.axvline(0, color=TXT, lw=0.4, alpha=0.5)
    ax3.set_xlabel("$\\Delta$R² (sensing added to personality)", fontsize=6)
    ax3.set_title("Incremental $\\Delta$R²\n(* = FDR < .05)",
                  fontsize=7, pad=8)
    clean_axis(ax3, grid_axis="x")

    fig.savefig(OUT / "fig2_core.png")
    plt.close(); print("  fig2 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 3 — FULL WIDTH (7.0 x 8) — Mechanism
# ═══════════════════════════════════════════════════════════════════════
def fig3_ieee():
    """Fig 3: SHAP + trait specificity + DL heatmap. Single column."""
    fig = plt.figure(figsize=(3.5, 7.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.9, 0.8, 1.2], hspace=0.45)

    # ── (a) SHAP heatmap ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
    tlbl = ["E", "A", "C", "N", "O"]
    files = {
        "S2-STAI": BY / "s2/tables/shap_personality_stai.csv",
        "S2-BAI":  BY / "s2/tables/shap_personality_bai.csv",
        "S3-BDI":  BY / "s3/tables/shap_personality_bdiii.csv",
        "S3-CESD": BY / "s3/tables/shap_personality_cesd.csv",
        "S3-PSS":  BY / "s3/tables/shap_personality_pss10.csv",
        "S3-STAI": BY / "s3/tables/shap_personality_stai.csv",
        "S3-UCLA": BY / "s3/tables/shap_personality_ucla.csv",
    }
    data = {}
    for lbl, fp in files.items():
        if not fp.exists(): continue
        d = pd.read_csv(fp)
        row = d[d.Model.str.contains("Ridge", case=False)]
        if len(row) == 0: row = d.iloc[:1]
        data[lbl] = [pd.to_numeric(row[t].values[0], errors="coerce")
                     if t in row.columns else 0 for t in traits]
    mat = pd.DataFrame(data, index=tlbl).T

    annot = mat.copy().astype(str)
    for i in range(mat.shape[0]):
        ranks = mat.iloc[i].rank(ascending=False).astype(int)
        for j in range(mat.shape[1]):
            star = " *" if ranks.iloc[j] == 1 else ""
            annot.iloc[i, j] = f"{mat.iloc[i, j]:.2f}{star}"

    cmap = LinearSegmentedColormap.from_list("shap",
        ["#FFF8F0", "#FDAE6B", "#D94801"])
    sns.heatmap(mat, annot=annot, fmt="", cmap=cmap, ax=ax,
                linewidths=1.0, linecolor=WHITE,
                cbar_kws={"label": "Mean |SHAP|", "shrink": 0.6,
                           "aspect": 20, "pad": 0.02},
                annot_kws={"fontsize": 6})
    ax.set_title("Big-Five SHAP Importance per Depression Scale (* = top predictor)",
                 fontsize=8, pad=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.5)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, fontweight="bold")

    # ── (b) Trait-domain specificity ─────────────────────────────────
    ax_b = fig.add_subplot(gs[1]); plabel(ax_b, "b")
    corr = pd.read_csv(CORE / "three_study_mh_correlations.csv")
    constructs = ["Depression", "State Anxiety", "Perceived Stress", "Loneliness"]
    trait_names = ["extraversion", "agreeableness", "conscientiousness",
                   "neuroticism", "openness"]
    trait_short = ["E", "A", "C", "N", "O"]

    corr_mat = np.full((len(constructs), len(trait_names)), np.nan)
    for i, con in enumerate(constructs):
        for j, tr in enumerate(trait_names):
            row = corr[(corr.Construct == con) & (corr.Trait == tr)]
            if len(row) == 0: continue
            r = row.iloc[0]
            vals = []
            for s in [1, 2, 3]:
                v = pd.to_numeric(r.get(f"S{s}_r", np.nan), errors="coerce")
                if pd.notna(v): vals.append(v)
            if vals:
                corr_mat[i, j] = np.mean(vals)

    gpa_corrs = pd.read_csv(Path("results/supplementary/personality_gpa_correlations.csv"))
    gpa_vals = []
    for tr in trait_names:
        row = gpa_corrs[gpa_corrs.Trait == tr]
        if len(row):
            vals = [pd.to_numeric(row.iloc[0].get(f"S{s}_r", np.nan), errors="coerce")
                    for s in [1, 2]]
            vals = [v for v in vals if pd.notna(v)]
            gpa_vals.append(np.mean(vals) if vals else np.nan)
        else:
            gpa_vals.append(np.nan)
    corr_mat = np.vstack([corr_mat, [gpa_vals]])
    constructs_ext = constructs + ["GPA"]

    corr_df = pd.DataFrame(corr_mat, index=constructs_ext, columns=trait_short)
    cmap_div = LinearSegmentedColormap.from_list("corrdiv",
        [RED, "#FFEEEE", WHITE, "#EEF0FF", BLU])
    annot_corr = corr_df.map(lambda x: f"{x:.2f}" if pd.notna(x) else "\u2014")
    sns.heatmap(corr_df, annot=annot_corr, fmt="", cmap=cmap_div, center=0,
                ax=ax_b, linewidths=1.0, linecolor=WHITE, vmin=-0.6, vmax=0.6,
                cbar_kws={"label": "Mean r (across studies)", "shrink": 0.6},
                annot_kws={"fontsize": 6.5})
    ax_b.set_title("Trait$\\rightarrow$Outcome Specificity (mean Pearson r across 3 studies)",
                   fontsize=8, pad=10)
    ax_b.set_yticklabels(ax_b.get_yticklabels(), rotation=0, fontsize=6.5)
    ax_b.set_xticklabels(ax_b.get_xticklabels(), fontsize=8, fontweight="bold")

    # ── (c) DL baseline heatmap ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[2]); plabel(ax2, "c")
    dl = pd.read_csv(ROB / "deep_learning_comparison.csv")

    model_order = ["Personality (Ridge)", "Sensing PCA (Ridge)",
                   "GradientBoosting+Optuna", "1D-CNN", "MOMENT \u2192 Ridge"]
    model_short = ["Personality\n(Ridge)", "Sensing\n(Ridge)", "Gradient\nBoosting",
                   "1D-CNN", "MOMENT"]

    outcomes = []
    for study in ["S2", "S3"]:
        sub = dl[dl.Study == study]
        for o in sub.Outcome.unique():
            outcomes.append(f"{study} {o}")

    mat_dl = np.full((len(outcomes), len(model_order)), np.nan)
    for i, (study, outcome) in enumerate(
            [(s, o) for s in ["S2", "S3"] for o in dl[dl.Study == s].Outcome.unique()]):
        for j, m in enumerate(model_order):
            row = dl[(dl.Study == study) & (dl.Outcome == outcome) & (dl.Model == m)]
            if len(row):
                val = row.R2_mean.values[0]
                mat_dl[i, j] = max(val, -0.2)

    cmap_r2 = LinearSegmentedColormap.from_list("r2div",
        [RED, "#FFEEEE", WHITE, "#EEF0FF", BLU])
    vmax = max(0.55, np.nanmax(mat_dl))

    im = ax2.imshow(mat_dl, cmap=cmap_r2, aspect="auto", vmin=-0.2, vmax=vmax, zorder=1)

    for i in range(mat_dl.shape[0]):
        for j in range(mat_dl.shape[1]):
            v = mat_dl[i, j]
            if np.isnan(v): continue
            actual = dl[(dl.Study == outcomes[i][:2]) &
                        (dl.Outcome == outcomes[i][3:]) &
                        (dl.Model == model_order[j])]
            if len(actual):
                real_v = actual.R2_mean.values[0]
                txt = f"{real_v:.2f}" if real_v > -1 else f"{real_v:.1f}"
            else:
                txt = f"{v:.2f}"
            color = WHITE if abs(v) > 0.3 else TXT
            ax2.text(j, i, txt, ha="center", va="center", fontsize=5,
                     fontweight="bold" if j == 0 else "normal", color=color)

    ax2.set_xticks(range(len(model_short)))
    ax2.set_xticklabels(model_short, fontsize=5.5)
    ax2.set_yticks(range(len(outcomes)))
    ax2.set_yticklabels(outcomes, fontsize=5.5)

    s2_count = len(dl[dl.Study == "S2"].Outcome.unique())
    ax2.axhline(s2_count - 0.5, color=WHITE, lw=1.5)

    cb = fig.colorbar(im, ax=ax2, shrink=0.6, pad=0.02, aspect=20)
    cb.set_label("Cross-Validated R²", fontsize=6)

    ax2.set_title("Only Personality Yields Positive R² across Five Architectures",
                  fontsize=8, pad=10)

    fig.savefig(OUT / "fig3_mechanism.png")
    plt.close(); print("  fig3 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 4 — SINGLE COLUMN (3.5 x 7) — Robustness
# ═══════════════════════════════════════════════════════════════════════
def fig4_ieee():
    """Fig 4: dose-response + ICC + transfer. Single column."""
    fig = plt.figure(figsize=(3.5, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.7], hspace=0.45)

    # ── (a) Dose-response ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a", x=-0.18, y=1.06)
    dr = pd.read_csv(ROB / "dose_response.csv")
    days_list = sorted(dr.N_days.unique())

    pers_mean = dr.groupby("Outcome").R2_personality.mean().mean()
    ax.axhline(pers_mean, color=BLU, ls="--", lw=0.8, zorder=2, alpha=0.8)
    ax.text(93, pers_mean + 0.004, f"Pers mean = {pers_mean:.2f}",
            fontsize=5, color=BLU, va="bottom", ha="right", alpha=0.8)

    sens_by_day = dr.groupby("N_days").R2_behavior.agg(["mean", "min", "max"])
    x_smooth = np.linspace(min(days_list), max(days_list), 200)
    try:
        spl_mean = make_interp_spline(days_list, sens_by_day["mean"].values, k=2)
        spl_lo = make_interp_spline(days_list, sens_by_day["min"].values, k=2)
        spl_hi = make_interp_spline(days_list, sens_by_day["max"].values, k=2)
        y_mean = spl_mean(x_smooth)
        y_lo = spl_lo(x_smooth)
        y_hi = spl_hi(x_smooth)
    except Exception:
        x_smooth = np.array(days_list)
        y_mean = sens_by_day["mean"].values
        y_lo = sens_by_day["min"].values
        y_hi = sens_by_day["max"].values

    ax.fill_between(x_smooth, y_lo, y_hi, color=RED, alpha=0.1, zorder=1,
                    label="Sensing range")
    ax.plot(x_smooth, y_mean, color=RED, lw=1.2, zorder=3, label="Sensing mean")
    ax.scatter(days_list, sens_by_day["mean"].values, color=RED, s=15, zorder=4,
              edgecolors=WHITE, lw=0.4)

    ax.axhline(0, color=TXT, lw=0.3, alpha=0.3)
    ax.set_ylim(-0.06, 0.18)
    ax.set_xlabel("Days of Sensing Data", fontsize=6)
    ax.set_ylabel("Cross-Validated R²", fontsize=6)
    ax.set_title("More Sensing Data Does Not Help", fontsize=7, pad=8)
    ax.set_xticks(days_list)
    ax.tick_params(axis='x', labelsize=5)
    ax.legend(fontsize=5, frameon=True, edgecolor="#DDDDDD",
              fancybox=False, borderpad=0.3, loc="upper left")
    clean_axis(ax, grid_axis="y")

    ax.text(0.98, 0.05, "7 days $\\approx$ 92 days",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, fontweight="bold", color=TXT, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec="#DDDDDD", lw=0.4))

    # ── (b) ICC ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b", x=-0.18, y=1.06)
    icc = pd.read_csv(ROB / "temporal_reliability.csv")
    keep = ["Steps", "Sleep Duration", "Screen Time", "Home Time"]
    icc = icc[icc.Feature.isin(keep)]

    markers_icc = {'Steps': 'o', 'Sleep Duration': 's', 'Screen Time': '^', 'Home Time': 'D'}
    alphas_icc = {'Steps': 1.0, 'Sleep Duration': 0.75, 'Screen Time': 0.6, 'Home Time': 0.45}

    for feat in keep:
        sub = icc[icc.Feature == feat].sort_values("Window_days")
        if len(sub) == 0: continue
        a = alphas_icc.get(feat, 0.7)
        xd = sub.Window_days.values
        yd = sub.ICC.values
        try:
            x_sm = np.linspace(xd.min(), xd.max(), 100)
            spl = make_interp_spline(xd, yd, k=2)
            y_sm = spl(x_sm)
            ax2.plot(x_sm, y_sm, color=RED, alpha=a, lw=0.8, zorder=2)
        except Exception:
            ax2.plot(xd, yd, color=RED, alpha=a, lw=0.8, zorder=2)
        ax2.scatter(xd, yd, color=RED, alpha=a, s=15, zorder=4,
                    marker=markers_icc.get(feat, 'o'),
                    edgecolors=WHITE, lw=0.4, label=feat)
        ax2.fill_between(sub.Window_days, sub.ICC_CI_lo, sub.ICC_CI_hi,
                         color=RED, alpha=a * 0.07, zorder=0)

    ax2.axhspan(0.75, 0.85, color=BLU, alpha=0.06, zorder=0)
    ax2.axhline(0.85, color=BLU, ls="--", lw=0.6, alpha=0.5)
    ax2.axhline(0.75, color=BLU, ls=":", lw=0.6, alpha=0.5)
    ax2.text(6, 0.865, "BFI-44 r = .85", fontsize=4.5, color=BLU, alpha=0.7)
    ax2.text(6, 0.735, "BFI-10 r = .75", fontsize=4.5, color=BLU, alpha=0.7)

    ax2.legend(fontsize=5, loc="center right", frameon=True, edgecolor="#DDDDDD",
               fancybox=False, borderpad=0.3)

    ax2.text(0.5, 0.08, "Stable but predicts nothing",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=6, fontweight="bold", color=TXT,
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDE7", ec="#E0D870",
                       lw=0.4, alpha=0.9))

    ax2.set_xlabel("Window Size (days)", fontsize=6)
    ax2.set_ylabel("ICC(3,k)", fontsize=6)
    ax2.set_title("Sensing Is Reliable but Irrelevant", fontsize=7, pad=8)
    ax2.set_xticks([7, 14, 30, 45, 60])
    ax2.set_xlim(3, 65)
    ax2.set_ylim(0.55, 1.0)
    clean_axis(ax2, grid_axis="y")

    # ── (c) Cross-study transfer ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[2]); plabel(ax3, "c", x=-0.18, y=1.08)
    tr = pd.read_csv(ROB / "cross_study_transfer.csv")
    labels_tr = tr.Transfer.values
    x_tr = np.arange(len(tr))
    w = 0.3
    ax3.bar(x_tr - w/2, tr.R2_within_study, w, color=BLU, edgecolor=WHITE, lw=0.4,
            label="Within-study", zorder=3, alpha=0.8)
    ax3.bar(x_tr + w/2, tr.R2_transfer, w, color=RED, edgecolor=WHITE, lw=0.4,
            label="Cross-study", zorder=3, alpha=0.8)
    for i in range(len(tr)):
        ax3.text(i - w/2, tr.R2_within_study.iloc[i] + 0.01,
                 f"{tr.R2_within_study.iloc[i]:.2f}", ha="center", va="bottom",
                 fontsize=4, color=BLU, fontweight="bold")
        ax3.text(i + w/2, min(tr.R2_transfer.iloc[i], 0) - 0.02,
                 f"{tr.R2_transfer.iloc[i]:.2f}", ha="center", va="top",
                 fontsize=4, color=RED, fontweight="bold")
    short_labels = [
        str(l).replace("Depression ", "").replace("Anxiety ", "")
              .replace("(CES-D)", "CESD").replace("(STAI)", "STAI")
              .replace("(", "").replace(")", "")
        for l in labels_tr
    ]
    ax3.set_xticks(x_tr)
    ax3.set_xticklabels(short_labels, fontsize=5, rotation=15, ha="right")
    ax3.axhline(0, color=TXT, lw=0.4, alpha=0.5)
    ax3.set_ylabel("Cross-Validated R²", fontsize=6)
    ax3.set_title("Personality Models Don't Transfer", fontsize=7, pad=8)
    ax3.legend(fontsize=5, frameon=True, edgecolor="#DDDDDD", fancybox=False)
    clean_axis(ax3, grid_axis="y")

    fig.savefig(OUT / "fig4_robustness.png")
    plt.close(); print("  fig4 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 5 — SINGLE COLUMN (3.5 x 7) — Clinical
# ═══════════════════════════════════════════════════════════════════════
def fig5_ieee():
    """Fig 5: NNS + cost + worst-20%. Single column."""
    fig = plt.figure(figsize=(3.5, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.48)

    # ── (a) NNS ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a", x=-0.18, y=1.06)
    nns = pd.read_csv(ROB / "nns_comparison_matched.csv")
    outcomes = nns.Outcome.unique()
    x = np.arange(len(outcomes))
    w = 0.3

    pers_vals, beh_vals = [], []
    for o in outcomes:
        p = nns[(nns.Outcome == o) & (nns.Features == "Pers-only")]
        b = nns[(nns.Outcome == o) & (nns.Features == "Beh-only")]
        pers_vals.append(p.TP_per_100.values[0] if len(p) else 0)
        beh_vals.append(b.TP_per_100.values[0] if len(b) else 0)

    ax.bar(x - w/2, pers_vals, w, color=BLU, edgecolor=WHITE, lw=0.4,
           label="Personality", zorder=3)
    ax.bar(x + w/2, beh_vals, w, color=RED, edgecolor=WHITE, lw=0.4,
           label="Sensing", zorder=3)

    for i in range(len(outcomes)):
        ax.text(i - w/2, pers_vals[i] + 0.3, f"{pers_vals[i]:.0f}", ha="center",
                va="bottom", fontsize=4.5, color=BLU, fontweight="bold")
        ax.text(i + w/2, beh_vals[i] + 0.3, f"{beh_vals[i]:.0f}", ha="center",
                va="bottom", fontsize=4.5, color=RED)
        delta = pers_vals[i] - beh_vals[i]
        if abs(delta) > 0.5:
            sign = "+" if delta > 0 else ""
            ymax = max(pers_vals[i], beh_vals[i]) + 1.8
            ax.text(i, ymax, f"{sign}{delta:.0f}", ha="center", fontsize=4,
                    color=BLU if delta > 0 else RED, fontweight="bold", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=4.5, rotation=20, ha="right")
    ax.set_ylabel("TP per 100 Screened", fontsize=6)
    ax.set_title("True Positives per 100\n(Matched N)", fontsize=7, pad=8)
    ax.legend(fontsize=5, frameon=True, edgecolor="#DDDDDD", fancybox=False,
              loc="upper right")
    clean_axis(ax, grid_axis="y")

    n_wins_p = sum(1 for p, b in zip(pers_vals, beh_vals) if p > b)
    badge(ax, f"Pers wins {n_wins_p}/{len(outcomes)}", 0.02, 0.92, color=BLU, fontsize=5)

    # ── (b) Cost-effectiveness scatter ───────────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b", x=-0.18, y=1.06)
    cost = pd.read_csv(CORE / "cost_effectiveness.csv")

    for _, row in cost.iterrows():
        is_p = any(k in row.Approach for k in ["BFI", "Neuroticism", "Big Five"])
        c = BLU if is_p else RED
        m = "o" if is_p else "s"
        ax2.scatter(row.Time_min, row.R2, color=c, marker=m, s=30, zorder=4,
                    edgecolors=WHITE, lw=0.5)

    offsets = {
        "2 BFI items (10 sec)": (6, 4),
        "Neuroticism score (1 min)": (6, -8),
        "Full Big Five (5 min)": (6, 4),
        "All 44 BFI items (5 min)": (6, -8),
        "Sensing PCA (weeks)": (-5, -10),
        "Sensing 28 raw (weeks)": (-5, 7),
        "Pers + Sensing PCA": (-5, 6),
        "Pers + Comm (S2 best)": (-5, -10),
    }
    for _, row in cost.iterrows():
        is_p = any(k in row.Approach for k in ["BFI", "Neuroticism", "Big Five"])
        c = BLU if is_p else RED
        off = offsets.get(row.Approach, (5, 4))
        ax2.annotate(row.Approach, (row.Time_min, row.R2),
                     textcoords="offset points", xytext=off,
                     fontsize=4, color=c, alpha=0.8,
                     arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.2)
                     if abs(off[0]) > 4 else None)

    ax2.axhline(0, color=TXT, lw=0.3, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xlabel("Assessment Time (min, log)", fontsize=6)
    ax2.set_ylabel("Cross-Validated R²", fontsize=6)
    ax2.set_title("Brief Questionnaire vs\nWeeks of Sensing", fontsize=7, pad=8)
    clean_axis(ax2, grid_axis="y")

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=BLU, markersize=4,
                  label='Questionnaire'),
           Line2D([0], [0], marker='s', color='w', markerfacecolor=RED, markersize=4,
                  label='Passive Sensing')]
    ax2.legend(handles=leg, frameon=True, edgecolor="#DDDDDD", fancybox=False,
               fontsize=5)

    # ── (c) Worst-20% rescue ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2]); plabel(ax3, "c", x=-0.18, y=1.08)
    w20 = pd.read_csv(ROB / "worst20_rescue.csv")
    x_w = np.arange(len(w20))
    bw = 0.18
    ax3.bar(x_w - 1.5*bw, w20.R2_pers_worst, bw, color=BLU, edgecolor=WHITE, lw=0.4,
            label="Pers (hard)", zorder=3, alpha=0.6)
    ax3.bar(x_w - 0.5*bw, w20.R2_beh_worst, bw, color=RED, edgecolor=WHITE, lw=0.4,
            label="Sens (hard)", zorder=3, alpha=0.6)
    ax3.bar(x_w + 0.5*bw, w20.R2_pers_rest, bw, color=BLU, edgecolor=WHITE, lw=0.4,
            label="Pers (easy)", zorder=3)
    ax3.bar(x_w + 1.5*bw, w20.R2_beh_rest, bw, color=RED, edgecolor=WHITE, lw=0.4,
            label="Sens (easy)", zorder=3)
    ax3.set_xticks(x_w)
    ax3.set_xticklabels(w20.Outcome, fontsize=5)
    ax3.axhline(0, color=TXT, lw=0.4, alpha=0.5)
    ax3.set_ylabel("Cross-Validated R²", fontsize=6)
    ax3.set_title("Sensing Doesn't Help\nPersonality's Hard Cases", fontsize=7, pad=8)
    ax3.legend(fontsize=4, frameon=True, edgecolor="#DDDDDD", fancybox=False,
               ncol=2, loc="upper right", borderpad=0.3)
    clean_axis(ax3, grid_axis="y")

    fig.savefig(OUT / "fig5_clinical.png")
    plt.close(); print("  fig5 done")


# ═══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES — IEEE sizes
# ═══════════════════════════════════════════════════════════════════════

def figS1_ieee():
    """Fig S1: Idiographic. Single column, vertical stack."""
    df = pd.read_csv(ROB / "idiographic_models.csv")
    r2 = df.R2_person.dropna()

    fig = plt.figure(figsize=(3.5, 5.2))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1.2, 1.0], hspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    plabel(ax1, "a"); plabel(ax2, "b"); plabel(ax3, "c")

    def _color_hist(ax_h, vals, bins):
        counts, edges, patches = ax_h.hist(vals, bins=bins, edgecolor=WHITE, lw=0.4, alpha=0.88)
        for patch, left in zip(patches, edges[:-1]):
            center = left + (edges[1] - edges[0]) / 2
            if center >= 0.3:
                patch.set_facecolor(BLU)
            elif center >= 0:
                patch.set_facecolor("#A8C4DB")
            else:
                patch.set_facecolor(GRY)

    _color_hist(ax1, r2.clip(-5, 1), bins=30)
    ax1.axvline(0, color=TXT, lw=0.6, ls="--", alpha=0.5)
    ax1.set_xlabel("Person-Level R²", fontsize=6)
    ax1.set_ylabel("Count", fontsize=6)
    ax1.set_title("Full Distribution", fontsize=7, pad=6)
    clean_axis(ax1, grid_axis="y")

    r2z = r2[r2 > -0.5]
    _color_hist(ax2, r2z, bins=25)
    ax2.axvline(0, color=TXT, lw=0.6, ls="--", alpha=0.5, label="R\u00b2 = 0")
    ax2.axvline(0.3, color=BLU, lw=0.6, ls="--", alpha=0.7, label="R\u00b2 = 0.3")
    ax2.axvspan(0.3, r2z.max() + 0.05, color=BLU, alpha=0.05)

    n03 = (r2 > 0.3).sum()
    n0 = (r2 > 0).sum()
    nt = len(r2)
    ax2.text(0.97, 0.95,
             f"{n03}/{nt} ({100*n03/nt:.0f}%) R\u00b2 > 0.3\n"
             f"{n0}/{nt} ({100*n0/nt:.0f}%) R\u00b2 > 0",
             transform=ax2.transAxes, ha="right", va="top", fontsize=5,
             bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec=BLU,
                       lw=0.4, alpha=0.9))
    ax2.set_xlabel("Person-Level R²", fontsize=6)
    ax2.set_title("Zoomed: R\u00b2 > -0.5", fontsize=7, pad=6)
    ax2.legend(fontsize=5, frameon=True, edgecolor="#DDDDDD", fancybox=False)
    clean_axis(ax2, grid_axis="y")

    wp = pd.read_csv(ROB / "within_person_correlation.csv")
    feat_short = {
        "steps": "Steps", "sleep_dur": "Sleep dur.", "sleep_eff": "Sleep eff.",
        "screen_dur": "Screen", "calls_in": "Calls in", "hometime": "Home time",
    }
    wp["short"] = wp.Feature.map(feat_short).fillna(wp.Feature)
    wp = wp.sort_values("Pct_strong", ascending=True)
    yc = np.arange(len(wp))
    pct = wp.Pct_strong.values * 100

    n_persons = wp.N_persons.values
    ax3.barh(yc, pct, color=BLU, edgecolor=WHITE, lw=0.4,
             height=0.6, alpha=0.85, zorder=3)
    for i, (p_val, n) in enumerate(zip(pct, n_persons)):
        ax3.text(p_val + 0.8, i, f"{p_val:.0f}% (N={int(n)})", va="center",
                 fontsize=4.5, color=TXT)

    ax3.axvline(20, color=TXT, lw=0.4, ls="--", alpha=0.4)
    ax3.set_yticks(yc)
    ax3.set_yticklabels(wp.short.values, fontsize=5.5)
    ax3.set_xlabel("% |within-person r| > .30", fontsize=6)
    ax3.set_title("Per-Feature Idiographic Signal", fontsize=7, pad=6)
    ax3.set_xlim(0, 55)
    clean_axis(ax3, grid_axis="x")

    # title in LaTeX caption instead

    fig.savefig(OUT / "figS1_idiographic.png")
    plt.close(); print("  figS1 done")


def figS2_ieee():
    """Fig S2: Cross-study + calibration + ablation. Full width."""
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = gridspec.GridSpec(3, 1, hspace=0.50)

    # ── (a) Cross-study grouped bar chart ────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    corr = pd.read_csv(CORE / "three_study_mh_correlations.csv")
    key_pairs = [
        ("neuroticism", "Depression", "Neu$\\rightarrow$Dep"),
        ("neuroticism", "State Anxiety", "Neu$\\rightarrow$Anx"),
        ("conscientiousness", "Depression", "Con$\\rightarrow$Dep"),
        ("extraversion", "Loneliness", "Ext$\\rightarrow$Lone"),
    ]

    bar_data = []
    for trait, construct, lbl in key_pairs:
        matches = corr[(corr.Trait.str.lower() == trait) &
                       (corr.Construct.str.contains(construct, case=False))]
        if len(matches) == 0: continue
        row = matches.iloc[0]
        vals = [pd.to_numeric(row[f"S{s}_r"], errors="coerce") for s in [1, 2, 3]]
        ci_los = [pd.to_numeric(row.get(f"S{s}_ci_lo", np.nan), errors="coerce") for s in [1, 2, 3]]
        ci_his = [pd.to_numeric(row.get(f"S{s}_ci_hi", np.nan), errors="coerce") for s in [1, 2, 3]]
        if any(pd.isna(v) for v in vals): continue
        bar_data.append((lbl, *vals, ci_los, ci_his))

    if bar_data:
        n_pairs = len(bar_data)
        x = np.arange(n_pairs)
        w = 0.22
        study_colors = [BLU, RED, GRY]
        study_labels = ["S1 (N=28)", "S2 (N=722)", "S3 (N=809)"]
        for si in range(3):
            vals = [bd[si + 1] for bd in bar_data]
            err_lo = [abs(bd[si + 1] - bd[4][si]) if pd.notna(bd[4][si]) else 0 for bd in bar_data]
            err_hi = [abs(bd[5][si] - bd[si + 1]) if pd.notna(bd[5][si]) else 0 for bd in bar_data]
            ax.bar(x + (si - 1) * w, vals, w, color=study_colors[si],
                   edgecolor=WHITE, lw=0.4, label=study_labels[si], zorder=3,
                   alpha=0.8, yerr=[err_lo, err_hi],
                   error_kw={"lw": 0.6, "capsize": 1.5, "capthick": 0.6, "color": TXT, "alpha": 0.6})

        ax.set_xticks(x)
        ax.set_xticklabels([bd[0] for bd in bar_data], fontsize=6)
        ax.axhline(0, color=TXT, lw=0.3, alpha=0.3)
        ax.set_ylabel("Correlation r", fontsize=6)
        ax.set_title("Key Associations Replicate Across Studies", fontsize=8, pad=10)
        ax.legend(fontsize=5, frameon=True, edgecolor="#DDDDDD",
                  fancybox=False, borderpad=0.4, ncol=3)
    clean_axis(ax, grid_axis="y")

    # ── (b) Calibration ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b")
    cal = pd.read_csv(ROB / "clinical_expanded.csv")
    outcomes = cal.Outcome.unique()
    x = np.arange(len(outcomes))
    w = 0.3

    pv, bv = [], []
    for o in outcomes:
        p = cal[(cal.Outcome == o) & (cal.Features == "Pers-only")]
        b = cal[(cal.Outcome == o) & (cal.Features == "Beh-only")]
        pv.append(p.Brier_mean.values[0] if len(p) else np.nan)
        bv.append(b.Brier_mean.values[0] if len(b) else np.nan)

    ax2.bar(x - w/2, pv, w, color=BLU, edgecolor=WHITE, lw=0.4, label="Personality",
            zorder=3)
    ax2.bar(x + w/2, bv, w, color=RED, edgecolor=WHITE, lw=0.4, label="Sensing",
            zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcomes, fontsize=5.5, rotation=15, ha="right")
    ax2.set_ylabel("Brier Score (lower = better)", fontsize=6)
    ax2.set_ylim(0, 0.32)
    ax2.set_title("Model Calibration (Brier Score)", fontsize=8, pad=10)
    ax2.legend(fontsize=5, frameon=True, edgecolor="#DDDDDD", fancybox=False)
    clean_axis(ax2, grid_axis="y")

    # ── (c) Ablation ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2]); plabel(ax3, "c")
    abl = pd.read_csv(ROB / "modality_ablation.csv")
    avg = abl.groupby("Modality").Delta_R2.mean().sort_values(ascending=True)
    cols = [BLU if v > 0 else RED for v in avg.values]
    ax3.barh(np.arange(len(avg)), avg.values, color=cols, edgecolor=WHITE,
             lw=0.4, height=0.55, zorder=3)
    ax3.set_yticks(np.arange(len(avg)))
    ax3.set_yticklabels(avg.index, fontsize=5.5)
    ax3.axvline(0, color=TXT, lw=0.4, alpha=0.5)
    ax3.set_xlabel("$\\Delta$R² (adding modality to personality)", fontsize=6)
    ax3.set_title("Per-Modality $\\Delta$R² (all |$\\Delta$R²| < 0.01)",
                  fontsize=8, pad=10)
    clean_axis(ax3, grid_axis="x")

    fig.savefig(OUT / "figS2_supplement.png")
    plt.close(); print("  figS2 done")


def figS3_ieee():
    """Fig S3: Item-level. Full width."""
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.55)

    df = pd.read_csv(ROB / "item_level_prediction.csv")
    cols = ["R2_2_best_items", "R2_3_best_items", "R2_neuroticism_only",
            "R2_full_big5", "R2_all_44_items", "R2_sensing_PCA", "R2_sensing_raw28"]
    labels = ["2 Best Items\n(10 sec)", "3 Best Items\n(15 sec)",
              "Neuroticism\n(1 min)", "Full Big Five\n(5 min)",
              "All 44 Items\n(5 min)", "Sensing PCA\n(weeks)",
              "Sensing Raw\n(weeks)"]
    colors = [BLU, BLU, BLU, BLU, BLU, RED, RED]

    # ── (a) Mean across outcomes ─────────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    means = [df[c].mean() for c in cols]
    y = np.arange(len(labels))
    ax.barh(y, means, color=colors, edgecolor=WHITE, lw=0.4,
            height=0.6, zorder=3, alpha=0.85)
    for i, v in enumerate(means):
        ha = "left" if v >= 0 else "right"
        offset = 0.01 if v >= 0 else -0.01
        ax.text(v + offset, i, f"{v:.3f}", va="center", ha=ha,
                fontsize=5, fontweight="bold", color=colors[i])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.axvline(0, color=TXT, lw=0.4, alpha=0.5)
    ax.set_xlabel("Mean R² (across CES-D, STAI, BAI)", fontsize=6)
    ax.set_title("Two Items Outperform Weeks of Sensing", fontsize=8, pad=10)
    clean_axis(ax, grid_axis="x")
    badge(ax, "10 seconds > 10 weeks", 0.62, 0.92, color=BLU, fontsize=6)

    # ── (b) Per-outcome heatmap ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b")
    short_cols = ["2 Items", "3 Items", "N only", "Big Five",
                  "44 Items", "Sens PCA", "Sens Raw"]
    mat = df[cols].copy()
    mat.columns = short_cols
    mat.index = df.Outcome
    cmap = LinearSegmentedColormap.from_list("r2",
        [RED, "#FFEEEE", WHITE, "#EEF0FF", BLU])
    annot_mat = mat.map(lambda x: f"{x:.2f}")
    sns.heatmap(mat, annot=annot_mat, fmt="", cmap=cmap, center=0,
                ax=ax2, linewidths=1.0, linecolor=WHITE,
                cbar_kws={"label": "R²", "shrink": 0.6},
                annot_kws={"fontsize": 6, "fontweight": "medium"}, vmin=-0.2, vmax=0.6)
    for text_obj in ax2.texts:
        try:
            val_str = text_obj.get_text()
            val = float(val_str)
            text_obj.set_color(WHITE if val >= 0.25 else TXT)
        except (TypeError, ValueError):
            text_obj.set_color(TXT)
    ax2.set_title("Per-Outcome Breakdown (S2)", fontsize=8, pad=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=6)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=5.5, rotation=20, ha="right")

    fig.savefig(OUT / "figS3_item_level.png")
    plt.close(); print("  figS3 done")


def figS4_ieee():
    """Fig S4: Literature benchmark. Full width."""
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    df = pd.read_csv(CORE / "literature_benchmark.csv")

    df = df[df.Metric == "AUC"].copy()
    df["Value"] = pd.to_numeric(df.Value, errors="coerce")
    df = df.dropna(subset=["Value"]).sort_values("Value")

    y = np.arange(len(df))
    colors = []
    for _, row in df.iterrows():
        if "This study" in row.Paper and "Pers-only" in row.Paper:
            colors.append(BLU)
        elif "This study" in row.Paper:
            colors.append(BLU if "Pers" in row.Paper else RED)
        else:
            colors.append(GRY)

    n_max = pd.to_numeric(df.N, errors="coerce").max() or 1
    for i, (_, row) in enumerate(df.iterrows()):
        c = colors[i]
        marker = "D" if "This study" in row.Paper else "o"
        n_val = pd.to_numeric(row.N, errors="coerce")
        if pd.notna(n_val) and n_val > 0:
            size = 20 + 100 * np.sqrt(n_val / n_max)
        else:
            size = 25
        if "This study" in row.Paper:
            size = max(size, 50)
        ax.scatter(row.Value, i, color=c, marker=marker, s=size, zorder=4,
                   edgecolors=WHITE, lw=0.5)

    labels = []
    for _, row in df.iterrows():
        n_str = f"N={int(row.N)}" if pd.notna(row.N) else ""
        labels.append(f"{row.Paper}  {n_str}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5)

    for i, (_, row) in enumerate(df.iterrows()):
        if "This study" in row.Paper:
            ax.axhspan(i - 0.4, i + 0.4, color=BLU, alpha=0.04, zorder=0)

    ax.axvline(0.5, color=TXT, lw=0.4, ls=":", alpha=0.3)
    ax.set_xlabel("AUC (marker size $\\propto \\sqrt{N}$)", fontsize=7)
    ax.set_xlim(0.45, 0.82)
    ax.set_title("Our Personality-Only Model vs Published Sensing Studies",
                 fontsize=8, pad=10)
    clean_axis(ax, grid_axis="x")

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='D', color='w', markerfacecolor=BLU,
                  markersize=5, label='This study'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=GRY,
                  markersize=4, label='Published work')]
    ax.legend(handles=leg, frameon=True, edgecolor="#DDDDDD", fancybox=False,
              fontsize=5.5, loc="lower right")

    fig.savefig(OUT / "figS4_literature.png")
    plt.close(); print("  figS4 done")


def figS5_ieee():
    """Fig S5: Rebuttals (4 panels). Full width, compact."""
    fig = plt.figure(figsize=(7.0, 3.8))
    gs = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.35)

    short_map = {
        "cesd_total": "CES-D", "stai_trait_total": "STAI-T", "bai_total": "BAI",
        "bdi2_total": "BDI-II", "stai_state": "STAI-S", "pss_10": "PSS",
        "ucla_loneliness": "UCLA",
    }

    # ── Panel a: Prospective ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0]); plabel(ax_a, "a")
    prosp = pd.read_csv(ROB / "prospective_prediction.csv")
    outcomes = prosp.Outcome.tolist()
    x = np.arange(len(outcomes))
    w = 0.32

    ar_vals = prosp.R2_autoregressive.values
    pers_delta = prosp.R2_auto_plus_pers.values - ar_vals
    sens_delta = prosp.R2_auto_plus_beh.values - ar_vals

    ax_a.bar(x - w/2, ar_vals, w, color=GRY, edgecolor=WHITE, lw=0.4,
             label="AR baseline", zorder=3, alpha=0.85)
    ax_a.bar(x - w/2, pers_delta, w, bottom=ar_vals, color=BLU, edgecolor=WHITE,
             lw=0.4, label="+ Pers \u0394R\u00b2", zorder=4, alpha=0.85)
    ax_a.bar(x + w/2, ar_vals, w, color=GRY, edgecolor=WHITE, lw=0.4,
             zorder=3, alpha=0.85)
    ax_a.bar(x + w/2, sens_delta, w, bottom=ar_vals, color=RED, edgecolor=WHITE,
             lw=0.4, label="+ Sens \u0394R\u00b2", zorder=4, alpha=0.85)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(outcomes, fontsize=5, rotation=15, ha="right")
    ax_a.set_ylabel("R\u00b2 (prospective)", fontsize=6)
    ax_a.axhline(0, color=TXT, lw=0.3, alpha=0.3)
    ax_a.set_ylim(-0.05, 0.70)
    ax_a.set_title("Prospective: AR Dominates", fontsize=7, pad=8)
    ax_a.legend(fontsize=4.5, frameon=True, edgecolor="#DDDDDD", fancybox=False,
                loc="upper right", borderpad=0.3)
    clean_axis(ax_a, grid_axis="y")
    ax_a.text(0.02, 0.97, "Both |\u0394R\u00b2| < .02",
              transform=ax_a.transAxes, ha="left", va="top", fontsize=5,
              color=TXT, alpha=0.75, style="italic")

    # ── Panel b: Demographic controls ────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1]); plabel(ax_b, "b")
    demo = pd.read_csv(Path("results/supplementary/demographic_controls.csv"))
    demo_outcomes = demo.Outcome.tolist()
    xb = np.arange(len(demo_outcomes))

    demo_base = demo.R2_gender.values
    pers_dr2 = demo.DR2_pers.values
    sens_dr2 = demo.DR2_beh.values

    ax_b.bar(xb - w/2, demo_base, w, color=GRY, edgecolor=WHITE, lw=0.4,
             label="Demographics", zorder=3, alpha=0.85)
    ax_b.bar(xb - w/2, pers_dr2, w, bottom=demo_base, color=BLU, edgecolor=WHITE,
             lw=0.4, label="+ Pers \u0394R\u00b2", zorder=4, alpha=0.85)
    ax_b.bar(xb + w/2, demo_base, w, color=GRY, edgecolor=WHITE, lw=0.4,
             zorder=3, alpha=0.85)
    ax_b.bar(xb + w/2, sens_dr2, w, bottom=demo_base, color=RED, edgecolor=WHITE,
             lw=0.4, label="+ Sens \u0394R\u00b2", zorder=4, alpha=0.85)

    for i, (dr_p, p_p) in enumerate(zip(demo.DR2_pers.values, demo.p_pers.values)):
        if p_p < 0.001:
            ax_b.text(i - w/2, demo_base[i] + dr_p + 0.012, "***",
                      ha="center", va="bottom", fontsize=6, color=BLU,
                      fontweight="bold")
    for i, (dr_b, p_b) in enumerate(zip(demo.DR2_beh.values, demo.p_beh.values)):
        if p_b >= 0.05:
            ax_b.text(i + w/2, demo_base[i] + max(dr_b, 0) + 0.015, "n.s.",
                      ha="center", va="bottom", fontsize=5, color=RED,
                      fontstyle="italic")

    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(demo_outcomes, fontsize=5.5)
    ax_b.set_ylabel("Cumulative R\u00b2", fontsize=6)
    ax_b.axhline(0, color=TXT, lw=0.3, alpha=0.3)
    ax_b.set_ylim(0, 0.75)
    ax_b.set_title("Demographics Controlled", fontsize=7, pad=8)
    ax_b.legend(fontsize=4.5, frameon=True, edgecolor="#DDDDDD", fancybox=False,
                loc="upper right", borderpad=0.3)
    clean_axis(ax_b, grid_axis="y")

    # ── Panel c: Disattenuation ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0]); plabel(ax_c, "c")
    dis = pd.read_csv(ROB / "disattenuation.csv")

    dis["label"] = dis.Study + " " + dis.Outcome.map(short_map).fillna(dis.Outcome)
    pers_rows = dis[dis.Predictor.str.startswith("Personality")]
    sens_rows = dis[dis.Predictor.str.startswith("Sensing")]

    sens_labels = set(sens_rows.label.values)
    keep = pers_rows[pers_rows.label.isin(sens_labels)].reset_index(drop=True)
    sens_keep = sens_rows.set_index("label").loc[keep.label.values].reset_index()

    yc = np.arange(len(keep))
    bar_h = 0.35

    pers_obs = keep.R2_observed.values
    pers_corr = keep.R2_corrected.values
    pers_incr = pers_corr - pers_obs

    sens_obs = sens_keep.R2_observed.values
    sens_corr = sens_keep.R2_corrected.values

    ax_c.barh(yc + bar_h/2, pers_obs, bar_h, color=BLU, edgecolor=WHITE, lw=0.4,
              alpha=0.45, zorder=3, label="Pers observed")
    ax_c.barh(yc + bar_h/2, pers_incr, bar_h, left=pers_obs, color=BLU,
              edgecolor=WHITE, lw=0.4, alpha=0.95, zorder=4,
              label="Pers correction")
    ax_c.barh(yc - bar_h/2, sens_corr, bar_h, color=RED, edgecolor=WHITE, lw=0.4,
              alpha=0.85, zorder=3, label="Sens corrected")

    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(keep.label.values, fontsize=5.5)
    ax_c.axvline(0, color=TXT, lw=0.4, alpha=0.5)
    ax_c.set_xlabel("R\u00b2 (observed \u2192 disattenuated)", fontsize=6)
    ax_c.set_title("Disattenuation: Gap Widens", fontsize=7, pad=8)
    ax_c.set_xlim(-0.1, 0.9)
    ax_c.legend(fontsize=4.5, frameon=True, edgecolor="#DDDDDD", fancybox=False,
                loc="lower right", borderpad=0.3)
    clean_axis(ax_c, grid_axis="x")
    ax_c.text(0.98, 0.97, "Sens \u03b1 = .93 (> BFI-10 \u03b1 = .65)",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=5,
              color=TXT, alpha=0.75, style="italic")

    # ── Panel d: Power analysis ──────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1]); plabel(ax_d, "d")
    pw = pd.read_csv(ROB / "power_analysis.csv")

    pw["label"] = pw.Study + " " + pw.Outcome.map(short_map).fillna(pw.Outcome)
    pw = pw.sort_values("Delta_R2", ascending=True).reset_index(drop=True)
    yp = np.arange(len(pw))

    for i, row in pw.iterrows():
        powered = row.Power_alpha05 >= 0.80
        c = BLU if powered else RED
        ax_d.scatter(row.Delta_R2, i, s=30, color=c, zorder=4,
                     edgecolors=WHITE, lw=0.4)
        ax_d.scatter(row.Min_DR2_80pct, i, s=25, facecolors="white",
                     edgecolors=TXT, lw=0.6, zorder=3, marker="o")
        x_lo = min(row.Delta_R2, row.Min_DR2_80pct)
        x_hi = max(row.Delta_R2, row.Min_DR2_80pct)
        ax_d.plot([x_lo, x_hi], [i, i], color="#CCCCCC", lw=0.6, zorder=2)

    ax_d.set_yticks(yp)
    ax_d.set_yticklabels(pw.label.values, fontsize=5.5)
    ax_d.axvline(0, color=TXT, lw=0.3, alpha=0.3)
    ax_d.set_xlabel("\u0394R\u00b2 (incremental sensing)", fontsize=6)
    ax_d.set_title("Power: 6/8 Adequately Powered", fontsize=7, pad=8)
    clean_axis(ax_d, grid_axis="x")

    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BLU,
               markersize=5, label='Observed (powered)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED,
               markersize=5, label='Observed (under-powered)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=TXT, markersize=5, label='Min @ 80%'),
    ]
    ax_d.legend(handles=leg, fontsize=4.5, frameon=True, edgecolor="#DDDDDD",
                fancybox=False, loc="lower right", borderpad=0.3)

    fig.suptitle("Robustness to Alternative Explanations",
                 fontsize=9, fontweight="bold", y=1.0)

    fig.savefig(OUT / "figS5_rebuttals.png")
    plt.close(); print("  figS5 done")


def figS6_ieee():
    """Fig S6: Clinical subgroup analysis. Full width."""
    sg = pd.read_csv(ROB / "subgroup_analysis.csv")
    fig = plt.figure(figsize=(7.0, 4.0))
    gs = gridspec.GridSpec(1, 2, wspace=0.28)

    outcomes = ["CES-D", "STAI", "BDI-II", "STAI-State", "PSS-10"]

    def _panel(ax, split_type, title):
        hi_rows = sg[(sg.Split_type == split_type) &
                     (sg.Subgroup.str.contains("High", na=False))]
        lo_rows = sg[(sg.Split_type == split_type) &
                     (sg.Subgroup.str.contains("Low", na=False))]

        def _val(df, outc, col):
            row = df[df.Outcome == outc]
            return row[col].values[0] if len(row) else np.nan

        x = np.arange(len(outcomes))
        w = 0.18

        pers_hi = [_val(hi_rows, o, "R2_personality") for o in outcomes]
        sens_hi = [_val(hi_rows, o, "R2_sensing")     for o in outcomes]
        pers_lo = [_val(lo_rows, o, "R2_personality") for o in outcomes]
        sens_lo = [_val(lo_rows, o, "R2_sensing")     for o in outcomes]

        ax.bar(x - 1.5*w, pers_hi, w, color=BLU, edgecolor=WHITE, lw=0.4,
               alpha=0.55, label="Pers (high)", zorder=3)
        ax.bar(x - 0.5*w, sens_hi, w, color=RED, edgecolor=WHITE, lw=0.4,
               alpha=0.55, label="Sens (high)", zorder=3)
        ax.bar(x + 0.5*w, pers_lo, w, color=BLU, edgecolor=WHITE, lw=0.4,
               alpha=0.95, label="Pers (low)", zorder=3)
        ax.bar(x + 1.5*w, sens_lo, w, color=RED, edgecolor=WHITE, lw=0.4,
               alpha=0.95, label="Sens (low)", zorder=3)

        n_strs = []
        dash = "\u2013"
        for o in outcomes:
            hi_n = _val(hi_rows, o, "N")
            lo_n = _val(lo_rows, o, "N")
            hi_s = str(int(hi_n)) if not np.isnan(hi_n) else dash
            lo_s = str(int(lo_n)) if not np.isnan(lo_n) else dash
            n_strs.append(f"{hi_s}/{lo_s}")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{o}\nN={n}" for o, n in zip(outcomes, n_strs)],
                           fontsize=5)
        ax.axhline(0, color=TXT, lw=0.4, alpha=0.5)
        ax.set_ylabel("Cross-Validated R\u00b2", fontsize=6)
        ax.set_ylim(-0.80, 0.45)
        ax.set_title(title, fontsize=7, pad=8)
        ax.legend(fontsize=4.5, frameon=True, edgecolor="gray",
                  facecolor="white", framealpha=1.0,
                  fancybox=False, loc="lower left", borderpad=0.3, ncol=2)
        clean_axis(ax, grid_axis="y")

    ax_a = fig.add_subplot(gs[0]); plabel(ax_a, "a")
    _panel(ax_a, "Clinical", "Clinical vs Subclinical")

    ax_b = fig.add_subplot(gs[1]); plabel(ax_b, "b")
    _panel(ax_b, "Neuroticism", "High vs Low Neuroticism")

    ax_a.text(0.02, 0.02,
              "Clinical: both fail;\nsensing drops further",
              transform=ax_a.transAxes, ha="left", va="bottom",
              fontsize=5, color=TXT, alpha=0.75, style="italic",
              bbox=dict(boxstyle="round,pad=0.2", fc="#FFF5F0", ec=RED,
                        lw=0.4, alpha=0.8))
    ax_b.text(0.02, 0.02,
              "Personality advantage\nholds at both levels",
              transform=ax_b.transAxes, ha="left", va="bottom",
              fontsize=5, color=TXT, alpha=0.75, style="italic",
              bbox=dict(boxstyle="round,pad=0.2", fc="#F0F5FF", ec=BLU,
                        lw=0.4, alpha=0.8))

    fig.suptitle("Does Personality's Advantage Generalize to Clinical Subsamples?",
                 fontsize=9, fontweight="bold", y=1.0)

    fig.savefig(OUT / "figS6_subgroup.png")
    plt.close(); print("  figS6 done")


# ═══════════════════════════════════════════════════════════════════════
ALL = {
    "fig1": fig1_ieee,
    "fig2": fig2_ieee,
    "fig3": fig3_ieee,
    "fig4": fig4_ieee,
    "fig5": fig5_ieee,
    "figS1": figS1_ieee,
    "figS2": figS2_ieee,
    "figS3": figS3_ieee,
    "figS4": figS4_ieee,
    "figS5": figS5_ieee,
    "figS6": figS6_ieee,
}

if __name__ == "__main__":
    setup_ieee()
    targets = sys.argv[1:] or list(ALL.keys())
    print(f"Generating {len(targets)} IEEE figures...", flush=True)
    for n in targets:
        if n in ALL:
            print(f"  [{n}]", end=" ", flush=True)
            ALL[n]()
        else:
            print(f"  Unknown: {n}")
    print(f"\nDone -> {OUT}/", flush=True)
