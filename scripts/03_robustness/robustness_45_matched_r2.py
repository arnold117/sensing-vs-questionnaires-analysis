#!/usr/bin/env python3
"""
Analysis 45 — Matched-N R² Sensitivity
=======================================
Re-run the grand synthesis Table 2 comparison on the intersection sample
(participants with BOTH personality and sensing data complete) so that
personality and sensing R² are computed on identical subsets.

Output: results/robustness/grand_synthesis_matched.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

OUT = Path("results/robustness")
OUT.mkdir(parents=True, exist_ok=True)
TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
RS = 42


def matched_cv_r2(X_pers, X_sens, y, n_splits=5, n_repeats=5, alpha=1.0):
    """Run Ridge CV on the INTERSECTION sample for both feature sets."""
    mask = (~np.isnan(X_pers).any(axis=1) &
            ~np.isnan(X_sens).any(axis=1) &
            ~np.isnan(y))
    X_p, X_s, y_c = X_pers[mask], X_sens[mask], y[mask]
    n = len(y_c)
    if n < 20:
        nan_res = {"R2_mean": np.nan, "R2_ci_lo": np.nan, "R2_ci_hi": np.nan}
        return nan_res, nan_res, n

    ns = min(n_splits, n // 3)
    if ns < 2:
        nan_res = {"R2_mean": np.nan, "R2_ci_lo": np.nan, "R2_ci_hi": np.nan}
        return nan_res, nan_res, n

    cv = RepeatedKFold(n_splits=ns, n_repeats=n_repeats, random_state=RS)
    r2_pers, r2_sens = [], []

    for tr, te in cv.split(X_p):
        sc = StandardScaler()
        # Personality
        Xtr_p = sc.fit_transform(X_p[tr])
        Xte_p = sc.transform(X_p[te])
        m_p = Ridge(alpha=alpha)
        m_p.fit(Xtr_p, y_c[tr])
        r2_pers.append(r2_score(y_c[te], m_p.predict(Xte_p)))
        # Sensing
        sc2 = StandardScaler()
        Xtr_s = sc2.fit_transform(X_s[tr])
        Xte_s = sc2.transform(X_s[te])
        m_s = Ridge(alpha=alpha)
        m_s.fit(Xtr_s, y_c[tr])
        r2_sens.append(r2_score(y_c[te], m_s.predict(Xte_s)))

    r2_pers, r2_sens = np.array(r2_pers), np.array(r2_sens)
    res_p = {"R2_mean": float(np.mean(r2_pers)),
             "R2_ci_lo": float(np.percentile(r2_pers, 2.5)),
             "R2_ci_hi": float(np.percentile(r2_pers, 97.5))}
    res_s = {"R2_mean": float(np.mean(r2_sens)),
             "R2_ci_lo": float(np.percentile(r2_sens, 2.5)),
             "R2_ci_hi": float(np.percentile(r2_sens, 97.5))}
    return res_p, res_s, n


def main():
    print("Loading datasets...", flush=True)
    s1 = pd.read_parquet("data/processed/analysis_dataset.parquet")
    s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
    s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")

    S1_BEH = [c for c in s1.columns if c.endswith("_pc1")]
    S2_BEH = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
    S3_BEH = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

    datasets = [
        ("S1", s1, S1_BEH, {
            "phq9_total": "PHQ-9", "pss_total": "PSS",
            "loneliness_total": "Loneliness", "gpa_overall": "GPA"
        }, 3, 3),
        ("S2", s2, S2_BEH, {
            "cesd_total": "CES-D", "stai_trait_total": "STAI-Trait", "bai_total": "BAI",
            "loneliness_total": "Loneliness", "self_esteem_total": "Self-Esteem",
            "gpa_overall": "GPA"
        }, 5, 5),
        ("S3", s3, S3_BEH, {
            "bdi2_total": "BDI-II", "stai_state": "STAI-State", "pss_10": "PSS-10",
            "cesd_total": "CESD", "ucla_loneliness": "UCLA"
        }, 5, 5),
    ]

    rows = []
    for study, df, beh_cols, outcomes, ns, nr in datasets:
        valid_pers = [c for c in TRAITS if c in df.columns]
        valid_beh = [c for c in beh_cols if c in df.columns]
        for col, label in outcomes.items():
            if col not in df.columns:
                continue
            res_p, res_s, n = matched_cv_r2(
                df[valid_pers].values, df[valid_beh].values, df[col].values,
                n_splits=ns, n_repeats=nr
            )
            rows.append({
                "Study": study, "Outcome": label, "N_matched": n,
                "R2_pers_matched": res_p["R2_mean"],
                "R2_pers_ci_lo": res_p["R2_ci_lo"],
                "R2_pers_ci_hi": res_p["R2_ci_hi"],
                "R2_sens_matched": res_s["R2_mean"],
                "R2_sens_ci_lo": res_s["R2_ci_lo"],
                "R2_sens_ci_hi": res_s["R2_ci_hi"],
                "Delta": res_p["R2_mean"] - res_s["R2_mean"] if not np.isnan(res_p["R2_mean"]) else np.nan,
                "Pers_wins": res_p["R2_mean"] > res_s["R2_mean"] if not np.isnan(res_p["R2_mean"]) else np.nan,
            })
            print(f"  {study} {label:15s}  N={n:4d}  "
                  f"Pers R²={res_p['R2_mean']:+.3f}  Sens R²={res_s['R2_mean']:+.3f}  "
                  f"Δ={res_p['R2_mean'] - res_s['R2_mean']:+.3f}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "grand_synthesis_matched.csv", index=False)
    print(f"\nSaved: {OUT / 'grand_synthesis_matched.csv'}")

    # Compare with unmatched
    orig = pd.read_csv(OUT / "grand_synthesis.csv")
    print("\n--- Matched vs Unmatched comparison ---")
    for _, row in result.iterrows():
        orig_row = orig[(orig.Study == row.Study) & (orig.Outcome == row.Outcome)]
        if len(orig_row) == 0:
            continue
        o = orig_row.iloc[0]
        print(f"  {row.Study} {row.Outcome:15s}  "
              f"Unmatched: Pers={o.R2_personality:+.3f} (N={int(o.N)}), "
              f"Matched: Pers={row.R2_pers_matched:+.3f} (N={row.N_matched})")

    wins = result["Pers_wins"].sum()
    total = result["Pers_wins"].notna().sum()
    print(f"\nMatched-N: Personality wins {int(wins)}/{total}")


if __name__ == "__main__":
    main()
