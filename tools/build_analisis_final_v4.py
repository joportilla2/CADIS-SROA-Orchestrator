# -*- coding: utf-8 -*-
"""
build_analisis_final_v4.py

Generador de Excel "paper-ready" para el prototipo CADIS SROA.
Versión 4: Corrige mapeo de columnas para McNemar y refina la búsqueda de archivos.

Este script NO recalcula simulaciones: solo consume los resultados ya
generados por el pipeline y los organiza en tablas listas para publicación.

Uso típico:
    python build_analisis_final_v4.py \
        --project-root . \
        --outdirs out_eta_f001 out_eta_f005 \
        --output-xlsx analisis_final_v4.xlsx \
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
    """Normaliza nombres de hoja para Excel (máx. 31 chars, sin caracteres prohibidos)."""
    bad_chars = '[]:*?/\\'
    out = ''.join('_' if c in bad_chars else c for c in name.strip())
    if not out:
        out = "Sheet"
    return out[:31]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Lee JSONL completo o hasta max_rows."""
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
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
# Estadísticos básicos (Rigurosos para Q1)
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
    p-valor exacto de McNemar (binomial 2-colas).
    Usa log-espacio para evitar overflow en factoriales grandes.
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
    """Odds-ratio pareada b/c."""
    if b == 0 and c == 0:
        return 1.0
    if c == 0:
        return np.inf
    return b / c


def paired_or_ci_log(b: int, c: int, alpha: float = 0.05) -> Tuple[float, float]:
    """CI aproximado log-normal para la OR pareada."""
    z = 1.96
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
    """CI bootstrap percentil."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return (np.nan, np.nan)
    n = vals.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.mean(vals[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha))


# --------------------------------------------------------------------
# PRCC y Ranking
# --------------------------------------------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ranking tipo Spearman con promedio en empates."""
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
    """Calcula PRCC usando rangos (Spearman parcial)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # Rank transform
    Xr = np.vstack([_rankdata(X[:, j]) for j in range(X.shape[1])]).T
    yr = _rankdata(y)

    # Centrar
    Xr -= Xr.mean(axis=0, keepdims=True)
    yr = yr - yr.mean()

    p = Xr.shape[1]
    out = np.full(p, np.nan, dtype=float)

    for i in range(p):
        cols = [j for j in range(p) if j != i]
        Z = Xr[:, cols]
        Z1 = np.column_stack([np.ones(Z.shape[0]), Z])

        xi = Xr[:, i]
        # Regresión parcial
        beta_x, *_ = np.linalg.lstsq(Z1, xi, rcond=None)
        rx = xi - Z1 @ beta_x

        beta_y, *_ = np.linalg.lstsq(Z1, yr, rcond=None)
        ry = yr - Z1 @ beta_y

        denom = (np.linalg.norm(rx) * np.linalg.norm(ry))
        out[i] = float((rx @ ry) / denom) if denom > 0 else np.nan

    return out


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Ajuste Benjamini-Hochberg FDR."""
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
    """PRCC + CI bootstrap + p-values por permutación."""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    n, p = X.shape

    if n < max(20, p + 5):
        # Muestra muy pequeña, solo cálculo directo
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
# Descubrimiento de archivos
# --------------------------------------------------------------------


@dataclass
class CompareRunArtifacts:
    outdir: str
    regime: str
    path: Path
    config_json: Optional[Path]
    run_summary_xlsx: Optional[Path]
    audit_jsonl: Optional[Path]


def discover_compare_runs(baseline_dir: Path, outdir_label: str,
                          missing: List[Dict[str, Any]]) -> List[CompareRunArtifacts]:
    runs: List[CompareRunArtifacts] = []
    if not baseline_dir.exists():
        missing.append({"where": str(baseline_dir), "what": "baseline_dir_missing", "severity": "high"})
        return runs

    for sub in sorted(baseline_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("compare_"):
            continue
        regime = sub.name.replace("compare_", "")
        
        # Archivos clave
        config = sub / "config.json"
        summary = sub / "run_summary.xlsx"
        audit = sub / "audit_cases.jsonl"

        runs.append(CompareRunArtifacts(
            outdir=outdir_label,
            regime=regime,
            path=sub,
            config_json=config if config.exists() else None,
            run_summary_xlsx=summary if summary.exists() else None,
            audit_jsonl=audit if audit.exists() else None,
        ))

        if not summary.exists():
            missing.append({"where": str(sub), "what": "run_summary_xlsx_missing", "severity": "high"})

    if not runs:
        missing.append({"where": str(baseline_dir), "what": "no_compare_runs_found", "severity": "high"})
    return runs


def discover_sensitivity_artifacts(sens_dir: Path, outdir_label: str,
                                  missing: List[Dict[str, Any]]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"outdir": outdir_label, "sens_dir": sens_dir}
    if not sens_dir.exists():
        missing.append({"where": str(sens_dir), "what": "sensitivity_dir_missing", "severity": "high"})
        return info

    # Metrics
    mx = sens_dir / "metrics.xlsx"
    mc = sens_dir / "metrics.csv"
    info["metrics_xlsx"] = mx if mx.exists() else None
    info["metrics_csv"] = mc if mc.exists() else None

    # Analysis folder (Pareto / PRCC)
    # IMPORTANTE: Esto solo existe si se corrió `cli.py analyze`
    analysis_dir = sens_dir / "analysis"
    prcc_files = []
    pareto_files = []
    
    if analysis_dir.exists():
        for reg_dir in sorted(analysis_dir.iterdir()):
            if not reg_dir.is_dir():
                continue
            regime = reg_dir.name
            
            # Pareto
            px = reg_dir / "pareto_front.xlsx"
            pc = reg_dir / "pareto_front.csv"
            if px.exists() or pc.exists():
                pareto_files.append({
                    "regime": regime,
                    "xlsx": px if px.exists() else None,
                    "csv": pc if pc.exists() else None
                })
            
            # PRCC exportado previamente
            prx = reg_dir / "prcc.xlsx"
            if prx.exists():
                prcc_files.append({"regime": regime, "xlsx": prx})
    else:
        # Advertencia amigable si falta la carpeta analysis
        missing.append({
            "where": str(sens_dir), 
            "what": "analysis_folder_missing", 
            "severity": "med",
            "fix": "Run 'python -m cadis_sroa2.cli analyze ...' to generate Pareto/PRCC files."
        })

    info["prcc_files"] = prcc_files
    info["pareto_files"] = pareto_files
    return info


# --------------------------------------------------------------------
# Construcción de tablas: BASELINE
# --------------------------------------------------------------------


def build_baseline_tables(compare_runs: List[CompareRunArtifacts],
                          n_boot: int,
                          rng: np.random.Generator,
                          missing: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    
    metrics_list = []
    mcnemar_list = []
    deltas_list = []
    audit_list = []
    repro_list = []

    for run in compare_runs:
        ctx = {"outdir": run.outdir, "regime": run.regime}
        
        # Repro index
        conf = {}
        if run.config_json:
            conf = read_json(run.config_json)
        repro_list.append({
            "outdir": run.outdir,
            "regime": run.regime,
            "seed": conf.get("seed"),
            "code_fp": conf.get("code_fingerprint"),
            "config_fp": conf.get("config_fingerprint")
        })

        if not run.run_summary_xlsx:
            continue

        try:
            xl = pd.ExcelFile(run.run_summary_xlsx)
            if "method_metrics" in xl.sheet_names:
                mm = xl.parse("method_metrics")
                mm.insert(0, "outdir", run.outdir)
                mm.insert(1, "regime", run.regime)
                metrics_list.append(mm)

                # --- FIX v4: Búsqueda robusta de b y c ---
                # Las simulaciones guardan: 'paired_hybrid_correct_only' (b) y 'paired_pure_correct_only' (c)
                # en la fila correspondiente al metodo 'hybrid' usualmente.
                
                # Buscamos la fila "hybrid"
                hyb_row = mm[mm["method"] == "hybrid"]
                if not hyb_row.empty:
                    row = hyb_row.iloc[0]
                    
                    # Posibles nombres de columna para b
                    b_opts = ["paired_hybrid_correct_only", "paired_b", "b"]
                    c_opts = ["paired_pure_correct_only", "paired_c", "c"]
                    
                    b_val = next((row[k] for k in b_opts if k in row and pd.notnull(row[k])), None)
                    c_val = next((row[k] for k in c_opts if k in row and pd.notnull(row[k])), None)
                    
                    if b_val is not None and c_val is not None:
                        b, c = int(b_val), int(c_val)
                        p_val = mcnemar_exact_pvalue(b, c)
                        mcnemar_list.append({
                            **ctx,
                            "b_hybrid_correct_pure_wrong": b,
                            "c_pure_correct_hybrid_wrong": c,
                            "total_discordant": b + c,
                            "p_value_exact": p_val,
                            "OR": paired_or(b, c)
                        })
                    else:
                        missing.append({**ctx, "what": "mcnemar_cols_not_found_in_hybrid_row", "severity": "high"})

                # Deltas (Hybrid - Pure)
                pure_row = mm[mm["method"] == "pure"]
                if not hyb_row.empty and not pure_row.empty:
                    h_row = hyb_row.iloc[0]
                    p_row = pure_row.iloc[0]
                    for m in ["accuracy", "macro_f1"]:
                        if m in h_row and m in p_row:
                            val_h = float(h_row[m])
                            val_p = float(p_row[m])
                            deltas_list.append({
                                **ctx,
                                "metric": m,
                                "pure": val_p,
                                "hybrid": val_h,
                                "delta": val_h - val_p
                            })

            if run.audit_jsonl:
                cases = read_jsonl(run.audit_jsonl)
                for c in cases:
                    c["outdir"] = run.outdir
                    c["regime"] = run.regime
                    audit_list.append(c)

        except Exception as e:
            missing.append({**ctx, "what": "error_reading_summary", "error": str(e), "severity": "high"})

    # Compilación
    df_metrics = pd.concat(metrics_list, ignore_index=True) if metrics_list else pd.DataFrame()
    df_mcnemar = pd.DataFrame(mcnemar_list)
    df_audit = pd.DataFrame(audit_list)
    df_repro = pd.DataFrame(repro_list)
    
    # CI Bootstrap para Deltas
    df_deltas_ci = pd.DataFrame()
    if deltas_list:
        raw_deltas = pd.DataFrame(deltas_list)
        ci_rows = []
        for (od, reg, met), sub in raw_deltas.groupby(["outdir", "regime", "metric"]):
            vals = sub["delta"].values
            lo, hi = (np.nan, np.nan)
            if len(vals) > 1:
                lo, hi = bootstrap_ci(vals, n_boot, rng)
            ci_rows.append({
                "outdir": od, "regime": reg, "metric": met,
                "mean_delta": vals.mean(),
                "ci_lo": lo, "ci_hi": hi,
                "n": len(vals)
            })
        df_deltas_ci = pd.DataFrame(ci_rows)

    return {
        "BASELINE_METRICS": df_metrics,
        "BASELINE_MCNEMAR": df_mcnemar,
        "BASELINE_DELTAS_CI": df_deltas_ci,
        "AUDIT_CASES": df_audit,
        "REPRODUCIBILITY": df_repro
    }


# --------------------------------------------------------------------
# Construcción de tablas: SENSITIVITY
# --------------------------------------------------------------------


def build_sensitivity_tables(sens_infos: List[Dict[str, Any]],
                             n_boot: int,
                             prcc_boot: int,
                             prcc_perm: int,
                             rng: np.random.Generator,
                             missing: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    
    all_metrics = []
    all_pareto = []

    for info in sens_infos:
        outdir = info["outdir"]
        
        # Leer metrics (CSV o XLSX)
        df = None
        if info["metrics_xlsx"]:
            df = pd.ExcelFile(info["metrics_xlsx"]).parse(0)
        elif info["metrics_csv"]:
            df = pd.read_csv(info["metrics_csv"])
        
        if df is not None:
            df["outdir"] = outdir
            all_metrics.append(df)

        # Leer Pareto
        for pfile in info["pareto_files"]:
            pdf = None
            if pfile["xlsx"]:
                pdf = pd.ExcelFile(pfile["xlsx"]).parse(0)
            elif pfile["csv"]:
                pdf = pd.read_csv(pfile["csv"])
            
            if pdf is not None:
                pdf["outdir"] = outdir
                pdf["regime"] = pfile["regime"]
                all_pareto.append(pdf)
            else:
                 missing.append({"where": str(info["sens_dir"]), "what": f"pareto_empty_{pfile['regime']}", "severity": "med"})

    # DataFrames combinados
    sens_combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    pareto_combined = pd.concat(all_pareto, ignore_index=True) if all_pareto else pd.DataFrame()

    sens_mcnemar = pd.DataFrame()
    prcc_infer = pd.DataFrame()

    if not sens_combined.empty:
        # 1. McNemar Pooled (SENS)
        # FIX v4: Buscar nombres correctos
        b_opts = ["paired_hybrid_correct_only", "paired_b", "b"]
        c_opts = ["paired_pure_correct_only", "paired_c", "c"]
        
        b_col = next((c for c in b_opts if c in sens_combined.columns), None)
        c_col = next((c for c in c_opts if c in sens_combined.columns), None)

        if b_col and c_col:
            rows = []
            for (od, reg), sub in sens_combined.groupby(["outdir", "regime"]):
                b_sum = sub[b_col].fillna(0).sum()
                c_sum = sub[c_col].fillna(0).sum()
                rows.append({
                    "outdir": od, "regime": reg,
                    "b_sum": b_sum, "c_sum": c_sum,
                    "p_exact": mcnemar_exact_pvalue(b_sum, c_sum),
                    "OR": paired_or(b_sum, c_sum)
                })
            sens_mcnemar = pd.DataFrame(rows)
        else:
            missing.append({"where": "sensitivity", "what": "mcnemar_cols_missing_in_combined", "cols": sens_combined.columns.tolist()})

        # 2. PRCC Calculation (Intra-group + Pooled)
        # Definir target
        target = "hybrid_accuracy" # Default target for PRCC usually
        if "delta_accuracy" not in sens_combined.columns and "pure_accuracy" in sens_combined.columns:
            sens_combined["delta_accuracy"] = sens_combined["hybrid_accuracy"] - sens_combined["pure_accuracy"]
            target = "delta_accuracy"
        
        params = [p for p in ["lam", "gamma", "omega", "forgetting", "drop_rate", "dup_rate"] if p in sens_combined.columns]

        if params:
            prcc_rows = []
            # Por grupo (Outdir x Regime)
            for (od, reg), sub in sens_combined.groupby(["outdir", "regime"]):
                X = sub[params].apply(pd.to_numeric, errors='coerce').fillna(0).values
                y = sub[target].apply(pd.to_numeric, errors='coerce').fillna(0).values
                
                # Check variance
                std = np.std(X, axis=0)
                valid_p = [params[i] for i in range(len(params)) if std[i] > 1e-9]
                
                if valid_p:
                    X_valid = sub[valid_p].values
                    inf = prcc_with_inference(X_valid, y, valid_p, prcc_boot, prcc_perm, rng)
                    inf["outdir"] = od
                    inf["regime"] = reg
                    inf["scope"] = "INTRA_GROUP"
                    prcc_rows.append(inf)

            # Pooled (Solo por Regimen, juntando outdirs)
            for reg, sub in sens_combined.groupby("regime"):
                X = sub[params].apply(pd.to_numeric, errors='coerce').fillna(0).values
                y = sub[target].values
                std = np.std(X, axis=0)
                valid_p = [params[i] for i in range(len(params)) if std[i] > 1e-9]
                
                if valid_p:
                    X_valid = sub[valid_p].values
                    inf = prcc_with_inference(X_valid, y, valid_p, prcc_boot, prcc_perm, rng)
                    inf["outdir"] = "POOLED"
                    inf["regime"] = reg
                    inf["scope"] = "POOLED"
                    prcc_rows.append(inf)
            
            if prcc_rows:
                prcc_infer = pd.concat(prcc_rows, ignore_index=True)

    return {
        "SENS_METRICS_RAW": sens_combined,
        "SENS_MCNEMAR_POOLED": sens_mcnemar,
        "PRCC_INFERENCE": prcc_infer,
        "PARETO_FRONTS": pareto_combined
    }


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Generador de tablas finales v4 (Corrección McNemar/Pareto)")
    ap.add_argument("--project-root", type=str, default=".", help="Raíz del proyecto")
    ap.add_argument("--outdirs", nargs="*", default=["out_eta_f001", "out_eta_f005"], help="Directorios de resultados")
    ap.add_argument("--output-xlsx", type=str, default="analisis_final_v4.xlsx", help="Archivo de salida")
    ap.add_argument("--seed", type=int, default=123, help="Semilla aleatoria")
    ap.add_argument("--n-boot", type=int, default=2000, help="Iteraciones bootstrap")
    
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    rng = np.random.default_rng(args.seed)
    missing = []

    # 1. Discovery
    compare_runs = []
    sens_infos = []
    
    for od in args.outdirs:
        p = root / od
        if not p.exists():
            missing.append({"what": f"outdir_not_found: {od}", "severity": "high"})
            continue
        
        compare_runs.extend(discover_compare_runs(p / "baseline", od, missing))
        sens_infos.append(discover_sensitivity_artifacts(p / "sensitivity", od, missing))

    # 2. Build Tables
    base_tables = build_baseline_tables(compare_runs, args.n_boot, rng, missing)
    sens_tables = build_sensitivity_tables(sens_infos, args.n_boot, 1000, 500, rng, missing)
    
    # 3. Overview & Missing
    overview_rows = []
    all_tables = {**base_tables, **sens_tables}
    for k, v in all_tables.items():
        if isinstance(v, pd.DataFrame):
            overview_rows.append({"sheet": k, "rows": len(v), "cols": v.shape[1]})
    
    all_tables["OVERVIEW"] = pd.DataFrame(overview_rows)
    all_tables["MISSING_WARNINGS"] = pd.DataFrame(missing)

    # 4. Write
    out_path = root / args.output_xlsx
    print(f"Escribiendo {out_path} ...")
    
    # Orden preferido de hojas
    order = [
        "OVERVIEW", 
        "BASELINE_METRICS", "BASELINE_MCNEMAR", "BASELINE_DELTAS_CI",
        "SENS_METRICS_RAW", "SENS_MCNEMAR_POOLED", "PRCC_INFERENCE", "PARETO_FRONTS",
        "AUDIT_CASES", "REPRODUCIBILITY", "MISSING_WARNINGS"
    ]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name in order:
            if name in all_tables:
                all_tables[name].to_excel(writer, sheet_name=safe_sheet_name(name), index=False)
    
    print("Done. Check MISSING_WARNINGS if sheets are empty.")
    if not all_tables["PARETO_FRONTS"].empty:
        print(" [OK] Pareto fronts found.")
    else:
        print(" [WARN] Pareto sheet empty. Did you run 'cli.py analyze'?")

    return 0

if __name__ == "__main__":
    main()