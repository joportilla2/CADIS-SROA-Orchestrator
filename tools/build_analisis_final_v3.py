# -*- coding: utf-8 -*-
"""
build_analisis_final_v3.py

Generador de Excel "paper-ready" para el prototipo cadis_sroa_rep2.

Este script NO recalcula simulaciones: solo consume los resultados ya
generados por el pipeline (out_eta_f001, out_eta_f005, etc.) y los
organiza en tablas listas para un artículo Q1 en Artificial Intelligence
in Medicine.

Uso típico (desde la raíz del proyecto):

    python tools/build_analisis_final_v3.py \
        --project-root . \
        --outdirs out_eta_f001 out_eta_f005 \
        --output-xlsx analisis_final_v3.xlsx \
        --n-bootstrap 4000 \
        --prcc-bootstrap 1500 \
        --prcc-perm 1000 \
        --seed 123
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# Utilidades generales
# --------------------------------------------------------------------


def safe_sheet_name(name: str) -> str:
    """Normaliza nombres de hoja para Excel (máx. 31 chars, sin []:*?/\)."""
    bad_chars = '[]:*?/\\'
    out = ''.join('_' if c in bad_chars else c for c in name.strip())
    if not out:
        out = "Sheet"
    return out[:31]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Lee JSONL completo o hasta max_rows (útil para auditoría ligera)."""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------
# Estadísticos básicos
# --------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Intervalo de Wilson para proporción k/n."""
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1.0 + (z ** 2) / n
    center = (phat + (z ** 2) / (2 * n)) / denom
    half = (z / denom) * math.sqrt(
        (phat * (1 - phat) / n) + ((z ** 2) / (4 * (n ** 2)))
    )
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """
    p-valor exacto de McNemar (binomial 2-colas) implementado en log‑espacio
    para evitar overflow: p = 2 * P(X <= min(b,c)), X ~ Bin(n, 0.5).
    """
    n = int(b) + int(c)
    if n <= 0:
        return 1.0

    k = int(min(b, c))
    log_half_pow_n = -n * math.log(2.0)

    def log_comb(nn: int, ii: int) -> float:
        return math.lgamma(nn + 1) - math.lgamma(ii + 1) - math.lgamma(nn - ii + 1)

    terms = [log_comb(n, i) + log_half_pow_n for i in range(k + 1)]
    m = max(terms)
    cdf = math.exp(m) * sum(math.exp(t - m) for t in terms)
    p = 2.0 * cdf
    return float(max(0.0, min(1.0, p)))


def paired_or(b: int, c: int) -> float:
    """Odds‑ratio pareada OR = b/c con manejo simple de ceros."""
    if b == 0 and c == 0:
        return 1.0
    if c == 0:
        return np.inf
    return b / c


def paired_or_ci_log(b: int, c: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    CI aproximado log‑normal para la OR pareada:
      log(OR) ± z * sqrt(1/b + 1/c)
    con corrección de Haldane‑Anscombe (+0.5) si hay ceros.
    """
    z = 1.96  # 95 %
    bb, cc = float(b), float(c)
    if bb == 0 or cc == 0:
        bb += 0.5
        cc += 0.5
    or_hat = bb / cc
    se = math.sqrt(1.0 / bb + 1.0 / cc)
    lo = math.exp(math.log(or_hat) - z * se)
    hi = math.exp(math.log(or_hat) + z * se)
    return float(lo), float(hi)


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                 ci: float = 0.95) -> Tuple[float, float]:
    """
    CI bootstrap percentil para una estadística pre‑calculada por muestra.
    Aquí usamos la media de `values` como estadístico.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    n = vals.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.mean(vals[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha))


# --------------------------------------------------------------------
# PRCC (Spearman parcial) + bootstrap + permutaciones
# --------------------------------------------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ranking tipo Spearman con promedio en empates (sin SciPy)."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks


def prcc_spearman(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcula PRCC estilo Spearman:
      1) rank(X) y rank(y)
      2) para cada parámetro i:
            Xi ~ X_{-i}, y ~ X_{-i}
      3) correlación Pearson de residuos.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xr = np.vstack([_rankdata(X[:, j]) for j in range(X.shape[1])]).T
    yr = _rankdata(y)

    # centrado
    Xr -= Xr.mean(axis=0, keepdims=True)
    yr = yr - yr.mean()

    p = Xr.shape[1]
    out = np.full(p, np.nan, dtype=float)

    for i in range(p):
        cols = [j for j in range(p) if j != i]
        Z = Xr[:, cols]
        Z1 = np.column_stack([np.ones(Z.shape[0]), Z])

        xi = Xr[:, i]
        beta_x, *_ = np.linalg.lstsq(Z1, xi, rcond=None)
        rx = xi - Z1 @ beta_x

        beta_y, *_ = np.linalg.lstsq(Z1, yr, rcond=None)
        ry = yr - Z1 @ beta_y

        denom = (np.linalg.norm(rx) * np.linalg.norm(ry))
        out[i] = float((rx @ ry) / denom) if denom > 0 else np.nan

    return out


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Ajuste Benjamini‑Hochberg FDR."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty_like(q)
    out[order] = q
    return out


def prcc_with_inference(X: np.ndarray, y: np.ndarray,
                        param_names: List[str],
                        n_boot: int,
                        n_perm: int,
                        rng: np.random.Generator) -> pd.DataFrame:
    """
    PRCC + CI bootstrap + p‑values por permutación (two‑sided).

    Devuelve columnas:
      param, prcc, ci_lo, ci_hi, p_perm, p_fdr_bh, n
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    n, p = X.shape
    if n < max(20, p + 5):
        # Muestra demasiado pequeña: reportamos solo PRCC sin inferencia.
        base = prcc_spearman(X, y) if n > p + 2 else np.full(p, np.nan)
        return pd.DataFrame({
            "param": param_names,
            "prcc": base,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p_perm": np.nan,
            "p_fdr_bh": np.nan,
            "n": n,
        })

    # PRCC base
    base = prcc_spearman(X, y)

    # Bootstrap
    boots = np.empty((n_boot, p), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b, :] = prcc_spearman(X[idx], y[idx])
    ci_lo = np.nanquantile(boots, 0.025, axis=0)
    ci_hi = np.nanquantile(boots, 0.975, axis=0)

    # Permutaciones
    exceed = np.zeros(p, dtype=int)
    for t in range(n_perm):
        yp = y.copy()
        rng.shuffle(yp)
        pr = prcc_spearman(X, yp)
        exceed += (np.abs(pr) >= np.abs(base)).astype(int)
    p_perm = (1.0 + exceed) / (1.0 + n_perm)
    p_fdr = bh_fdr(p_perm)

    return pd.DataFrame({
        "param": param_names,
        "prcc": base,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_perm": p_perm,
        "p_fdr_bh": p_fdr,
        "n": n,
    })


# --------------------------------------------------------------------
# Descubrimiento de artefactos: baseline + sensibilidad
# --------------------------------------------------------------------


@dataclass
class CompareRunArtifacts:
    outdir: str
    regime: str
    path: Path
    config_json: Optional[Path]
    checksums_json: Optional[Path]
    run_metrics_csv: Optional[Path]
    run_summary_xlsx: Optional[Path]
    audit_jsonl: Optional[Path]


def discover_compare_runs(baseline_dir: Path, outdir_label: str,
                          missing: List[Dict[str, Any]]) -> List[CompareRunArtifacts]:
    """
    Busca subdirectorios compare_* en baseline/ y registra sus artefactos.
    """
    runs: List[CompareRunArtifacts] = []
    if not baseline_dir.exists():
        missing.append({"where": str(baseline_dir), "what": "baseline_dir_missing", "severity": "high"})
        return runs

    for sub in sorted(baseline_dir.iterdir()):
        if not sub.is_dir() or not sub.name.lower().startswith("compare_"):
            continue
        regime = sub.name[len("compare_"):]
        config_json = sub / "config.json"
        checksums_json = sub / "checksums.json"
        run_metrics_csv = sub / "run_metrics.csv"
        run_summary_xlsx = sub / "run_summary.xlsx"
        audit_jsonl = sub / "audit_cases.jsonl"

        def opt(p: Path) -> Optional[Path]:
            return p if p.exists() else None

        runs.append(CompareRunArtifacts(
            outdir=outdir_label,
            regime=regime,
            path=sub,
            config_json=opt(config_json),
            checksums_json=opt(checksums_json),
            run_metrics_csv=opt(run_metrics_csv),
            run_summary_xlsx=opt(run_summary_xlsx),
            audit_jsonl=opt(audit_jsonl),
        ))

        if not run_summary_xlsx.exists():
            missing.append({"where": str(sub), "what": "run_summary_missing", "severity": "high"})

    if not runs:
        missing.append({"where": str(baseline_dir), "what": "no_compare_runs_found", "severity": "high"})
    return runs


def discover_sensitivity_artifacts(sens_dir: Path, outdir_label: str,
                                  missing: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Localiza metrics.(xlsx|csv), design.(xlsx|csv) y análisis por régimen
    (prcc.*, pareto_front.*) dentro de sensitivity/.
    """
    info: Dict[str, Any] = {"outdir": outdir_label, "sens_dir": sens_dir}
    if not sens_dir.exists():
        missing.append({"where": str(sens_dir), "what": "sensitivity_dir_missing", "severity": "high"})
        return info

    def pick(main: Path, alt: Path) -> Optional[Path]:
        if main.exists():
            return main
        if alt.exists():
            return alt
        return None

    info["metrics_xlsx"] = pick(sens_dir / "metrics.xlsx", sens_dir / "metrics.xls")
    info["metrics_csv"] = (sens_dir / "metrics.csv") if (sens_dir / "metrics.csv").exists() else None
    info["design_xlsx"] = pick(sens_dir / "design.xlsx", sens_dir / "design.xls")
    info["design_csv"] = (sens_dir / "design.csv") if (sens_dir / "design.csv").exists() else None

    analysis_dir = sens_dir / "analysis"
    info["analysis_dir"] = analysis_dir if analysis_dir.exists() else None

    prcc_files: List[Dict[str, Any]] = []
    pareto_files: List[Dict[str, Any]] = []
    if analysis_dir.exists():
        for reg_dir in sorted(analysis_dir.iterdir()):
            if not reg_dir.is_dir():
                continue
            regime = reg_dir.name
            prcc_files.append({
                "outdir": outdir_label,
                "regime": regime,
                "prcc_xlsx": (reg_dir / "prcc.xlsx") if (reg_dir / "prcc.xlsx").exists() else None,
                "prcc_csv": (reg_dir / "prcc.csv") if (reg_dir / "prcc.csv").exists() else None,
            })
            pareto_files.append({
                "outdir": outdir_label,
                "regime": regime,
                "pareto_xlsx": (reg_dir / "pareto_front.xlsx") if (reg_dir / "pareto_front.xlsx").exists() else None,
                "pareto_csv": (reg_dir / "pareto_front.csv") if (reg_dir / "pareto_front.csv").exists() else None,
            })
    info["prcc_files"] = prcc_files
    info["pareto_files"] = pareto_files
    return info


# --------------------------------------------------------------------
# Lectura de run_summary y helpers
# --------------------------------------------------------------------


def read_run_summary_xlsx(path: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    return {name: xl.parse(name) for name in xl.sheet_names}


def extract_method_metrics(sheets: Dict[str, pd.DataFrame],
                           missing: List[Dict[str, Any]],
                           ctx: Dict[str, Any]) -> Optional[pd.DataFrame]:
    if "method_metrics" not in sheets:
        missing.append({**ctx, "what": "method_metrics_sheet_missing", "severity": "high"})
        return None
    return sheets["method_metrics"].copy()


# --------------------------------------------------------------------
# Sensibilidad: utilidades
# --------------------------------------------------------------------


def robust_summary(df: pd.DataFrame, group_cols: List[str],
                   value_cols: List[str]) -> pd.DataFrame:
    """
    Resumen robusto: n, mean, std, median, q25, q75, p05, p95, min, max.
    """
    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for col in value_cols:
            x = pd.to_numeric(sub[col], errors="coerce").dropna().values
            if x.size == 0:
                stats = dict(
                    n=0, mean=np.nan, std=np.nan, median=np.nan,
                    q25=np.nan, q75=np.nan, p05=np.nan, p95=np.nan,
                    min=np.nan, max=np.nan,
                )
            else:
                stats = dict(
                    n=int(x.size),
                    mean=float(np.mean(x)),
                    std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                    median=float(np.median(x)),
                    q25=float(np.quantile(x, 0.25)),
                    q75=float(np.quantile(x, 0.75)),
                    p05=float(np.quantile(x, 0.05)),
                    p95=float(np.quantile(x, 0.95)),
                    min=float(np.min(x)),
                    max=float(np.max(x)),
                )
            for k, v in stats.items():
                base[f"{col}__{k}"] = v
        rows.append(base)
    return pd.DataFrame(rows)


def add_noise_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye bins para drop_rate y dup_rate:
      0, (0,0.1], ..., (0.4,0.5], (0.5,1.0]
    """
    out = df.copy()

    def bin_series(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        bins = [-1e-12, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        labels = ["0", "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-1.0"]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)

    out["drop_bin"] = bin_series(out["drop_rate"]) if "drop_rate" in out.columns else "NA"
    out["dup_bin"] = bin_series(out["dup_rate"]) if "dup_rate" in out.columns else "NA"
    return out


# --------------------------------------------------------------------
# Construcción de tablas baseline
# --------------------------------------------------------------------


def build_baseline_tables(compare_runs: List[CompareRunArtifacts],
                          n_boot: int,
                          rng: np.random.Generator,
                          missing: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    metrics_rows: List[pd.DataFrame] = []
    mcnemar_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    repro_rows: List[Dict[str, Any]] = []

    for run in compare_runs:
        ctx = {"outdir": run.outdir, "regime": run.regime, "run_path": str(run.path)}

        cfg = read_json(run.config_json) if run.config_json is not None else {}
        chk = read_json(run.checksums_json) if run.checksums_json is not None else {}

        repro_rows.append({
            "outdir": run.outdir,
            "regime": run.regime,
            "run_path": str(run.path),
            "has_config": bool(cfg),
            "has_checksums": bool(chk),
            "config_fingerprint": cfg.get("config_fingerprint"),
            "code_fingerprint": cfg.get("code_fingerprint"),
        })

        if run.run_summary_xlsx is None:
            missing.append({**ctx, "what": "run_summary_missing", "severity": "high"})
            continue

        sheets = read_run_summary_xlsx(run.run_summary_xlsx)
        mm = extract_method_metrics(sheets, missing, ctx)
        if mm is None:
            continue

        mm = mm.copy()
        mm.insert(0, "outdir", run.outdir)
        mm.insert(1, "regime", run.regime)
        mm.insert(2, "run_path", str(run.path))
        metrics_rows.append(mm)

        # Auditoría
        if run.audit_jsonl is not None:
            try:
                aud = read_jsonl(run.audit_jsonl, max_rows=None)
                for rec in aud:
                    r2 = dict(rec)
                    r2["outdir"] = run.outdir
                    r2["regime"] = run.regime
                    r2["run_path"] = str(run.path)
                    audit_rows.append(r2)
            except Exception as e:  # pragma: no cover - robustez
                missing.append({**ctx, "what": "audit_read_error", "error": str(e), "severity": "med"})

        # McNemar: buscamos columnas b/c o equivalentes
        cols_l = {str(c).lower(): c for c in mm.columns}
        b_col = next((cols_l.get(c) for c in ["paired_b", "mcnemar_b", "b"]), None)
        c_col = next((cols_l.get(c) for c in ["paired_c", "mcnemar_c", "c"]), None)

        if b_col and c_col:
            b_vals = pd.to_numeric(mm[b_col], errors="coerce").dropna()
            c_vals = pd.to_numeric(mm[c_col], errors="coerce").dropna()
            if not b_vals.empty and not c_vals.empty:
                b = int(b_vals.iloc[0])
                c = int(c_vals.iloc[0])
                p = mcnemar_exact_pvalue(b, c)
                or_hat = paired_or(b, c)
                or_lo, or_hi = paired_or_ci_log(b, c)
                mcnemar_rows.append({
                    **ctx,
                    "b_hybrid_correct_pure_wrong": b,
                    "c_pure_correct_hybrid_wrong": c,
                    "n_discordant": b + c,
                    "mcnemar_p_exact": p,
                    "paired_OR_b_over_c": or_hat,
                    "paired_OR_ci_lo": or_lo,
                    "paired_OR_ci_hi": or_hi,
                })
            else:
                missing.append({**ctx, "what": "no_numeric_b_c", "severity": "med"})
        else:
            missing.append({**ctx, "what": "missing_b_c_for_mcnemar", "severity": "high"})

        # Deltas híbrido – puro para accuracy y macro_f1
        def get_metric(mname: str, metric: str) -> Optional[float]:
            if "method" not in mm.columns or metric not in mm.columns:
                return None
            sub = mm[mm["method"].astype(str).str.lower() == mname]
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            return float(vals.mean()) if not vals.empty else None

        for metric in ["accuracy", "macro_f1"]:
            p_val = get_metric("pure", metric)
            h_val = get_metric("hybrid", metric)
            if p_val is not None and h_val is not None:
                delta_rows.append({
                    **ctx,
                    "metric": metric,
                    "pure": p_val,
                    "hybrid": h_val,
                    "delta_hybrid_minus_pure": h_val - p_val,
                })
            else:
                missing.append({**ctx, "what": f"missing_{metric}_for_delta", "severity": "med"})

    baseline_metrics = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    baseline_mcnemar = pd.DataFrame(mcnemar_rows)
    baseline_deltas = pd.DataFrame(delta_rows)

    # CI bootstrap sobre replicaciones por (outdir, regime, metric)
    ci_rows: List[Dict[str, Any]] = []
    if not baseline_deltas.empty:
        for (outdir, regime, metric), sub in baseline_deltas.groupby(["outdir", "regime", "metric"], dropna=False):
            vals = sub["delta_hybrid_minus_pure"].values.astype(float)
            if len(vals) >= 2:
                lo, hi = bootstrap_ci(vals, n_boot=n_boot, rng=rng)
            else:
                lo, hi = (np.nan, np.nan)
            ci_rows.append({
                "outdir": outdir,
                "regime": regime,
                "metric": metric,
                "delta_mean": float(np.mean(vals)),
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "n_rep": int(len(vals)),
            })
    baseline_deltas_ci = pd.DataFrame(ci_rows)
    audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame()
    repro_df = pd.DataFrame(repro_rows)

    return {
        "BASELINE_METRICS": baseline_metrics,
        "BASELINE_MCNEMAR": baseline_mcnemar,
        "BASELINE_DELTAS_CI": baseline_deltas_ci,
        "AUDIT_CASES_ALL": audit_df,
        "REPRODUCIBILITY_INDEX": repro_df,
    }


# --------------------------------------------------------------------
# Construcción de tablas de sensibilidad
# --------------------------------------------------------------------


def read_sens_metrics(info: Dict[str, Any],
                      missing: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    mx = info.get("metrics_xlsx")
    mc = info.get("metrics_csv")
    if mx is None and mc is None:
        missing.append({"where": str(info.get("sens_dir")), "what": "metrics_missing", "severity": "high"})
        return None
    if mx is not None:
        xl = pd.ExcelFile(mx)
        sheet = "metrics" if "metrics" in xl.sheet_names else xl.sheet_names[0]
        return xl.parse(sheet)
    return pd.read_csv(mc)


def build_sensitivity_tables(sens_infos: List[Dict[str, Any]],
                             n_boot: int,
                             prcc_boot: int,
                             prcc_perm: int,
                             rng: np.random.Generator,
                             missing: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    all_metrics: List[pd.DataFrame] = []
    pareto_all: List[pd.DataFrame] = []
    prcc_export: List[pd.DataFrame] = []

    for info in sens_infos:
        outdir = info["outdir"]
        dfm = read_sens_metrics(info, missing)
        if dfm is not None:
            dfm = dfm.copy()
            dfm.insert(0, "outdir", outdir)
            all_metrics.append(dfm)

        # Pareto
        for item in info.get("pareto_files", []):
            regime = item["regime"]
            px = item.get("pareto_xlsx")
            pc = item.get("pareto_csv")
            if px is None and pc is None:
                continue
            try:
                if px is not None:
                    xl = pd.ExcelFile(px)
                    dfp = xl.parse(xl.sheet_names[0])
                else:
                    dfp = pd.read_csv(pc)
                dfp = dfp.copy()
                dfp.insert(0, "outdir", outdir)
                dfp.insert(1, "regime", str(regime))
                pareto_all.append(dfp)
            except Exception as e:  # pragma: no cover
                missing.append({"outdir": outdir, "regime": regime, "what": "pareto_read_error", "error": str(e), "severity": "med"})

        # PRCC exportado (si existe)
        for item in info.get("prcc_files", []):
            regime = item["regime"]
            px = item.get("prcc_xlsx")
            pc = item.get("prcc_csv")
            if px is None and pc is None:
                continue
            try:
                if px is not None:
                    xl = pd.ExcelFile(px)
                    dfp = xl.parse(xl.sheet_names[0])
                else:
                    dfp = pd.read_csv(pc)
                dfp = dfp.copy()
                dfp.insert(0, "outdir", outdir)
                dfp.insert(1, "regime", str(regime))
                prcc_export.append(dfp)
            except Exception as e:  # pragma: no cover
                missing.append({"outdir": outdir, "regime": regime, "what": "prcc_read_error", "error": str(e), "severity": "med"})

    sens_combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    pareto_combined = pd.concat(pareto_all, ignore_index=True) if pareto_all else pd.DataFrame()
    prcc_export_df = pd.concat(prcc_export, ignore_index=True) if prcc_export else pd.DataFrame()

    sens_summary = pd.DataFrame()
    sens_mcnemar = pd.DataFrame()
    sens_bins = pd.DataFrame()
    prcc_infer = pd.DataFrame()

    if not sens_combined.empty:
        # Normalización ligera de nombres de columnas
        ctx = {"where": "sensitivity.metrics"}
        colmap: Dict[str, str] = {}
        pairs = [
            ("pure_accuracy", ["pure_accuracy", "pure_acc", "acc_pure"]),
            ("hybrid_accuracy", ["hybrid_accuracy", "hybrid_acc", "acc_hybrid"]),
            ("pure_macro_f1", ["pure_macro_f1", "pure_f1", "macro_f1_pure"]),
            ("hybrid_macro_f1", ["hybrid_macro_f1", "hybrid_f1", "macro_f1_hybrid"]),
        ]
        for canonical, opts in pairs:
            for c in opts:
                if c in sens_combined.columns:
                    colmap[canonical] = c
                    break

        if "pure_accuracy" in colmap and "hybrid_accuracy" in colmap:
            sens_combined["delta_accuracy"] = (
                pd.to_numeric(sens_combined[colmap["hybrid_accuracy"]], errors="coerce")
                - pd.to_numeric(sens_combined[colmap["pure_accuracy"]], errors="coerce")
            )
        if "pure_macro_f1" in colmap and "hybrid_macro_f1" in colmap:
            sens_combined["delta_macro_f1"] = (
                pd.to_numeric(sens_combined[colmap["hybrid_macro_f1"]], errors="coerce")
                - pd.to_numeric(sens_combined[colmap["pure_macro_f1"]], errors="coerce")
            )

        # Resumen robusto
        value_cols = [c for c in ["delta_accuracy", "delta_macro_f1",
                                  "filter_ratio_mean", "avg_filtered_attacks",
                                  "defeats_total_mean", "rad_mean"]
                      if c in sens_combined.columns]
        if value_cols:
            sens_summary = robust_summary(sens_combined, ["outdir", "regime"], value_cols)

        # McNemar pooled sobre sensibilidad
        b_col = next((c for c in ["paired_b", "mcnemar_b", "b"] if c in sens_combined.columns), None)
        c_col = next((c for c in ["paired_c", "mcnemar_c", "c"] if c in sens_combined.columns), None)
        if b_col and c_col:
            rows: List[Dict[str, Any]] = []
            for (outdir, regime), sub in sens_combined.groupby(["outdir", "regime"], dropna=False):
                b = int(pd.to_numeric(sub[b_col], errors="coerce").fillna(0).sum())
                c = int(pd.to_numeric(sub[c_col], errors="coerce").fillna(0).sum())
                p = mcnemar_exact_pvalue(b, c)
                or_hat = paired_or(b, c)
                or_lo, or_hi = paired_or_ci_log(b, c)
                rows.append({
                    "outdir": outdir,
                    "regime": regime,
                    "b_hybrid_correct_pure_wrong_sum": b,
                    "c_pure_correct_hybrid_wrong_sum": c,
                    "n_discordant_sum": b + c,
                    "mcnemar_p_exact": p,
                    "paired_OR_b_over_c": or_hat,
                    "paired_OR_ci_lo": or_lo,
                    "paired_OR_ci_hi": or_hi,
                })
            sens_mcnemar = pd.DataFrame(rows)
        else:
            missing.append({**ctx, "what": "missing_b_c_for_mcnemar_sensitivity", "severity": "high"})

        # Bins de ruido
        sens_b = add_noise_bins(sens_combined)
        bin_values = [c for c in ["delta_accuracy", "delta_macro_f1"] if c in sens_b.columns]
        if bin_values:
            sens_bins = robust_summary(
                sens_b,
                ["outdir", "regime", "drop_bin", "dup_bin"],
                bin_values,
            )

        # PRCC interno (intra‑grupo) usando target delta_accuracy o delta_macro_f1
        target = "delta_accuracy" if "delta_accuracy" in sens_combined.columns else                  ("delta_macro_f1" if "delta_macro_f1" in sens_combined.columns else None)
        param_order = [p for p in ["lam", "gamma", "omega", "forgetting", "drop_rate", "dup_rate"]
                       if p in sens_combined.columns]

        prcc_rows: List[pd.DataFrame] = []
        prcc_missing_const: List[Dict[str, Any]] = []

        if target and param_order:
            for (outdir, regime), sub in sens_combined.groupby(["outdir", "regime"], dropna=False):
                # Detectar parámetros constantes dentro del grupo
                X_all = sub[param_order].apply(pd.to_numeric, errors="coerce").values
                y_all = pd.to_numeric(sub[target], errors="coerce").values
                stds = np.nanstd(X_all, axis=0)
                varying_mask = stds > 0
                varying_params = [p for p, m in zip(param_order, varying_mask) if m]
                const_params = [p for p, m in zip(param_order, varying_mask) if not m]

                if const_params:
                    prcc_missing_const.append({
                        "outdir": outdir,
                        "regime": regime,
                        "where": "PRCC_intra",
                        "what": "constant_low",
                        "severity": "low",
                        "excluded": ",".join(const_params),
                        "reason": "Sin variación dentro del grupo => PRCC indefinido en este grupo",
                    })

                if not varying_params:
                    continue  # no hay nada que modelar

                X = sub[varying_params].apply(pd.to_numeric, errors="coerce").values
                y = y_all
                dfp = prcc_with_inference(
                    X=X,
                    y=y,
                    param_names=varying_params,
                    n_boot=prcc_boot,
                    n_perm=prcc_perm,
                    rng=rng,
                )
                dfp.insert(0, "outdir", outdir)
                dfp.insert(1, "regime", regime)
                dfp.insert(2, "target_metric", target)
                prcc_rows.append(dfp)

            if prcc_rows:
                prcc_infer = pd.concat(prcc_rows, ignore_index=True)

            # Inyectamos advertencias por constantes
            missing.extend(prcc_missing_const)
        else:
            missing.append({"where": "PRCC", "what": "no_target_or_params", "severity": "high"})

        # PRCC global (pooled sobre todos los outdirs para cada régimen)
        if target and param_order:
            pooled_rows: List[pd.DataFrame] = []
            pooled_missing_const: List[Dict[str, Any]] = []
            for regime, sub in sens_combined.groupby("regime", dropna=False):
                X_all = sub[param_order].apply(pd.to_numeric, errors="coerce").values
                y_all = pd.to_numeric(sub[target], errors="coerce").values
                stds = np.nanstd(X_all, axis=0)
                varying_mask = stds > 0
                varying_params = [p for p, m in zip(param_order, varying_mask) if m]
                const_params = [p for p, m in zip(param_order, varying_mask) if not m]

                if const_params:
                    pooled_missing_const.append({
                        "outdir": "POOLED",
                        "regime": regime,
                        "where": "PRCC_pooled",
                        "what": "constant_low",
                        "severity": "low",
                        "excluded": ",".join(const_params),
                        "reason": "Aún sin variación (incluso pooled) => PRCC indefinido para esos parámetros",
                    })

                if not varying_params:
                    continue

                X = sub[varying_params].apply(pd.to_numeric, errors="coerce").values
                y = y_all
                dfp = prcc_with_inference(
                    X=X,
                    y=y,
                    param_names=varying_params,
                    n_boot=prcc_boot,
                    n_perm=prcc_perm,
                    rng=rng,
                )
                dfp.insert(0, "outdir", "POOLED")
                dfp.insert(1, "regime", regime)
                dfp.insert(2, "target_metric", target)
                pooled_rows.append(dfp)

            if pooled_rows:
                prcc_pooled = pd.concat(pooled_rows, ignore_index=True)
                # concatenamos a prcc_infer para tener todo junto
                if not prcc_infer.empty:
                    prcc_infer = pd.concat([prcc_infer, prcc_pooled], ignore_index=True)
                else:
                    prcc_infer = prcc_pooled

            missing.extend(pooled_missing_const)

    # Resumen de Pareto
    pareto_summary = pd.DataFrame()
    if not pareto_combined.empty:
        rows: List[Dict[str, Any]] = []
        for (outdir, regime), sub in pareto_combined.groupby(["outdir", "regime"], dropna=False):
            rows.append({
                "outdir": outdir,
                "regime": regime,
                "pareto_points": int(len(sub)),
            })
        pareto_summary = pd.DataFrame(rows)

    return {
        "SENS_METRICS_COMBINED": sens_combined,
        "SENS_SUMMARY_ROBUST": sens_summary,
        "SENS_MCNEMAR_POOLED": sens_mcnemar,
        "SENS_BINS_NOISE": sens_bins,
        "PRCC_WITH_CI_FDR": prcc_infer,
        "PRCC_EXPORTED_RAW": prcc_export_df,
        "PARETO_FRONTS": pareto_combined,
        "PARETO_SUMMARY": pareto_summary,
    }


# --------------------------------------------------------------------
# OVERVIEW
# --------------------------------------------------------------------


def build_overview(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            rows.append({"sheet": name, "rows": int(df.shape[0]), "cols": int(df.shape[1])})
    return pd.DataFrame(rows).sort_values("sheet")


# --------------------------------------------------------------------
# main()
# --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".",
                    help="Raíz del proyecto (donde están out_eta_f001, out_eta_f005, etc.)")
    ap.add_argument("--outdirs", nargs="*", default=["out_eta_f001", "out_eta_f005"],
                    help="Outdirs a procesar (por defecto: out_eta_f001 out_eta_f005).")
    ap.add_argument("--output-xlsx", type=str, default="analisis_final_v3.xlsx",
                    help="Nombre del archivo Excel a generar.")
    ap.add_argument("--n-bootstrap", type=int, default=4000,
                    help="Bootstrap para CI de deltas.")
    ap.add_argument("--prcc-bootstrap", type=int, default=1500,
                    help="Bootstrap para CI de PRCC.")
    ap.add_argument("--prcc-perm", type=int, default=1000,
                    help="Permutaciones para p‑values de PRCC.")
    ap.add_argument("--seed", type=int, default=123,
                    help="Semilla global del análisis.")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    rng = np.random.default_rng(args.seed)

    missing: List[Dict[str, Any]] = []
    compare_runs_all: List[CompareRunArtifacts] = []
    sens_infos: List[Dict[str, Any]] = []

    # Descubrimos baseline + sensibilidad para cada outdir
    for outdir_name in args.outdirs:
        outdir_path = (project_root / outdir_name).resolve()
        if not outdir_path.exists():
            missing.append({"where": str(outdir_path), "what": "outdir_missing", "severity": "high"})
            continue
        baseline_dir = outdir_path / "baseline"
        sens_dir = outdir_path / "sensitivity"
        compare_runs_all.extend(discover_compare_runs(baseline_dir, outdir_name, missing))
        sens_infos.append(discover_sensitivity_artifacts(sens_dir, outdir_name, missing))

    tables: Dict[str, pd.DataFrame] = {}

    # Baseline
    tables.update(build_baseline_tables(compare_runs_all, args.n_bootstrap, rng, missing))

    # Sensibilidad
    tables.update(build_sensitivity_tables(sens_infos, args.n_bootstrap,
                                           args.prcc_bootstrap, args.prcc_perm, rng, missing))

    # MISSING_WARNINGS + OVERVIEW
    tables["MISSING_WARNINGS"] = pd.DataFrame(missing)
    tables["OVERVIEW"] = build_overview(tables)

    # Escritura Excel
    out_xlsx = (project_root / args.output_xlsx).resolve()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        preferred = [
            "OVERVIEW",
            "BASELINE_METRICS",
            "BASELINE_DELTAS_CI",
            "BASELINE_MCNEMAR",
            "SENS_METRICS_COMBINED",
            "SENS_SUMMARY_ROBUST",
            "SENS_MCNEMAR_POOLED",
            "SENS_BINS_NOISE",
            "PRCC_WITH_CI_FDR",
            "PARETO_SUMMARY",
            "PARETO_FRONTS",
            "AUDIT_CASES_ALL",
            "REPRODUCIBILITY_INDEX",
            "PRCC_EXPORTED_RAW",
            "MISSING_WARNINGS",
        ]
        written = set()
        for name in preferred:
            df = tables.get(name)
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=safe_sheet_name(name), index=False)
                written.add(name)
        # Cualquier hoja extra
        for name, df in tables.items():
            if name in written or not isinstance(df, pd.DataFrame):
                continue
            df.to_excel(writer, sheet_name=safe_sheet_name(name), index=False)

    print(f"[OK] Excel generado en: {out_xlsx}")
    if tables["MISSING_WARNINGS"].shape[0] > 0:
        print(f"[WARN] Se registraron {tables['MISSING_WARNINGS'].shape[0]} advertencias/faltantes. "
              "Revisar hoja MISSING_WARNINGS para transparencia.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
