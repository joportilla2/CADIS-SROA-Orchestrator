from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

from typing import Any, Dict, List, Tuple, Optional
import os
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .design import latin_hypercube, pareto_front
from .simulation import run_compare, run_compare_metrics_only
from .utils import write_json, sha256_text, stable_json, fingerprint_source, compute_checksums
from .excelio import save_workbook

# -------------------------
# Internal helpers
# -------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return
    cols = list(rows[0].keys())
    extra = sorted({k for r in rows for k in r.keys()} - set(cols))
    cols = cols + extra
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _write_design(outdir: str, design_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    design_csv = os.path.join(outdir, "design.csv")
    _write_csv(design_csv, design_rows)
    design_xlsx = os.path.join(outdir, "design.xlsx")
    save_workbook(design_xlsx, {"design": {"type": "dict_rows", "rows": design_rows}})
    return design_csv, design_xlsx

def _write_metrics(outdir: str, metrics_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    metrics_csv = os.path.join(outdir, "metrics.csv")
    _write_csv(metrics_csv, metrics_rows)
    metrics_xlsx = os.path.join(outdir, "metrics.xlsx")
    save_workbook(metrics_xlsx, {"metrics": {"type": "dict_rows", "rows": metrics_rows}})
    return metrics_csv, metrics_xlsx

def _code_fingerprint() -> str:
    here = os.path.dirname(__file__)
    sources = [os.path.join(here, f) for f in [
        "cli.py","pipeline.py","simulation.py","sroa.py","argumentation.py","agents.py","reputation.py","metrics.py","protocol.py","design.py","utils.py","excelio.py"
    ]]
    return fingerprint_source(sources)

# -------------------------
# Sensitivity analysis math
# -------------------------

def _rank(x: np.ndarray) -> np.ndarray:
    # average ranks for ties
    temp = x.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(x), dtype=float)
    # tie handling
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for c in np.where(counts > 1)[0]:
        idx = np.where(inv == c)[0]
        ranks[idx] = ranks[idx].mean()
    return ranks

def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    # residualize x and y w.r.t z then correlate
    # add intercept
    Z = np.column_stack([np.ones(len(z)), z])
    beta_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ beta_x
    ry = y - Z @ beta_y
    denom = (np.std(rx) * np.std(ry))
    if denom == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def compute_prcc(params: np.ndarray, y: np.ndarray, param_names: List[str]) -> Dict[str, float]:
    """PRCC on rank-transformed variables."""
    prcc: Dict[str, float] = {}
    # rank transform
    P = np.column_stack([_rank(params[:, i]) for i in range(params.shape[1])])
    yy = _rank(y)
    for i, name in enumerate(param_names):
        others = np.delete(P, i, axis=1)
        prcc[name] = _partial_corr(P[:, i], yy, others)
    return prcc

# -------------------------
# Pipeline steps
# -------------------------

def run_sensitivity(
    outdir: str,
    regimes: List[str],
    seeds: List[int],
    n_samples: int,
    warmup: int,
    test: int,
    lam_min: float,
    lam_max: float,
    gamma_min: float,
    gamma_max: float,
    omega_min: float,
    omega_max: float,
    design_seed: int,
    forgetting: float,
    coalition_size: int,
    include_adversary: bool,
    drop_rate: float,
    dup_rate: float,
    log_every_runs: int = 10,
    log_every_cases: int = 0,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Run LHS sensitivity across (lam,gamma,omega) for each regime and seed.

    v5: avoids per-run SQLite/JSONL. Produces:
      sensitivity/design.(csv,xlsx)
      sensitivity/metrics.(csv,xlsx)
      sensitivity/runs/*.json (config + checksums per run)
    """
    sens_dir = os.path.join(outdir, "sensitivity")
    _ensure_dir(sens_dir)
    runs_dir = os.path.join(sens_dir, "runs")
    _ensure_dir(runs_dir)

    # LHS in [0,1]^3
    lhs = latin_hypercube(n_samples, 3, seed=design_seed)
    design_rows: List[Dict[str, Any]] = []
    samples: List[Tuple[int, float, float, float]] = []
    for i in range(n_samples):
        lam = lam_min + (lam_max - lam_min) * float(lhs[i, 0])
        gam = gamma_min + (gamma_max - gamma_min) * float(lhs[i, 1])
        omg = omega_min + (omega_max - omega_min) * float(lhs[i, 2])
        samples.append((i, lam, gam, omg))
        design_rows.append({"sample_id": i, "lam": lam, "gamma": gam, "omega": omg})

    design_csv, design_xlsx = _write_design(sens_dir, design_rows)

    code_fp = _code_fingerprint()

    # Build jobs
    jobs: List[Dict[str, Any]] = []
    for reg in regimes:
        for (sid, lam, gam, omg) in samples:
            for seed in seeds:
                jobs.append({
                    "regime": reg,
                    "sample_id": sid,
                    "seed": seed,
                    "lam": lam,
                    "gamma": gam,
                    "omega": omg,
                })

    def _job_worker(job: Dict[str, Any]) -> Dict[str, Any]:
        # regime affects scenario overrides (kept stable with v4 naming)
        scenario_overrides = _regime_overrides(job["regime"])
        res = run_compare_metrics_only(
            seed=int(job["seed"]),
            warmup=warmup,
            test=test,
            lam=float(job["lam"]),
            gamma=float(job["gamma"]),
            omega=float(job["omega"]),
            forgetting=forgetting,
            coalition_size=coalition_size,
            include_adversary=include_adversary,
            drop_rate=drop_rate,
            dup_rate=dup_rate,
            scenario_overrides=scenario_overrides,
            audit_max_cases=0,  # sensitivity: no per-run audit by default
            audit_policy="diff",
            log_every_cases=log_every_cases,
        )
        row = {
            "regime": job["regime"],
            "sample_id": int(job["sample_id"]),
            "seed": int(job["seed"]),
            "lam": float(job["lam"]),
            "gamma": float(job["gamma"]),
            "omega": float(job["omega"]),
            "forgetting": float(forgetting),
            "eta_equivalent": float(1.0 - forgetting),
            "drop_rate": float(drop_rate),
            "dup_rate": float(dup_rate),
            # metrics
            "pure_accuracy": float(res["pure_accuracy"]),
            "hybrid_accuracy": float(res["hybrid_accuracy"]),
            "pure_macro_f1": float(res["pure_macro_f1"]),
            "hybrid_macro_f1": float(res["hybrid_macro_f1"]),
            "paired_pure_correct_only": int(res["paired_pure_correct_only"]),
            "paired_hybrid_correct_only": int(res["paired_hybrid_correct_only"]),
            "paired_both_correct": int(res["paired_both_correct"]),
            "paired_both_wrong": int(res["paired_both_wrong"]),
            "avg_message_count": float(res["avg_message_count"]),
            "avg_attacks_total": float(res["avg_attacks_total"]),
            "avg_filtered_attacks": float(res["avg_filtered_attacks"]),
            "avg_defeats_total": float(res["avg_defeats_total"]),
            "filter_ratio_mean": float(res["filter_ratio_mean"]),
            "EPR_sroa_hybrid": None if res["EPR_sroa_hybrid"] is None else float(res["EPR_sroa_hybrid"]),
            "CCR_sroa_hybrid": None if res["CCR_sroa_hybrid"] is None else float(res["CCR_sroa_hybrid"]),
            "rad_mean": float(res["rad_mean"]),
            "rad_p50": float(res["rad_p50"]),
            "rad_p95": float(res["rad_p95"]),
        }
        # write per-run config + checksums (small, auditable)
        run_id = f"{job['regime']}_s{int(job['sample_id']):04d}_seed{int(job['seed'])}"
        cfg = {
            "author": "Omar Portilla Jaimes",
            "email": "jorge.portilla2@unipamplona.edu.co",
            "run_id": run_id,
            "regime": job["regime"],
            "seed": int(job["seed"]),
            "warmup": warmup,
            "test": test,
            "lam": float(job["lam"]),
            "gamma": float(job["gamma"]),
            "omega": float(job["omega"]),
            "forgetting_rate": float(forgetting),
            "eta_equivalent": float(1.0 - forgetting),
            "coalition_size": int(coalition_size),
            "include_adversary": bool(include_adversary),
            "drop_rate": float(drop_rate),
            "dup_rate": float(dup_rate),
            "scenario_overrides": scenario_overrides,
        }
        cfg_fp = sha256_text(stable_json(cfg))
        cfg_out = os.path.join(runs_dir, f"{run_id}_config.json")
        write_json(cfg_out, {**cfg, "code_fingerprint": code_fp, "config_fingerprint": cfg_fp})
        chk_out = os.path.join(runs_dir, f"{run_id}_checksums.json")
        checks = compute_checksums([cfg_out], root=runs_dir)
        write_json(chk_out, checks)
        return row

    metrics_rows: List[Dict[str, Any]] = []
    total = len(jobs)

    if max_workers and max_workers > 1:
        # Windows-safe: spawn mode by default; functions are top-level.
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_job_worker, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                metrics_rows.append(fut.result())
                done += 1
                if log_every_runs and (done % log_every_runs == 0 or done == total):
                    print(f"[sensitivity] {done}/{total} runs")
    else:
        for j, job in enumerate(jobs, start=1):
            metrics_rows.append(_job_worker(job))
            if log_every_runs and (j % log_every_runs == 0 or j == total):
                print(f"[sensitivity] {j}/{total} runs")

    metrics_csv, metrics_xlsx = _write_metrics(sens_dir, metrics_rows)

    return {
        "sensitivity_dir": sens_dir,
        "design_csv": design_csv,
        "design_xlsx": design_xlsx,
        "metrics_csv": metrics_csv,
        "metrics_xlsx": metrics_xlsx,
        "n_runs": total,
    }

def analyze_sensitivity(outdir: str, regimes: List[str]) -> Dict[str, Any]:
    """Compute PRCC + Pareto front per regime and write Excel/CSV outputs."""
    sens_dir = os.path.join(outdir, "sensitivity")
    metrics_csv = os.path.join(sens_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Missing {metrics_csv}. Run sensitivity first.")

    import csv
    rows: List[Dict[str, Any]] = []
    with open(metrics_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    results: Dict[str, Any] = {"per_regime": {}}
    param_names = ["lam", "gamma", "omega"]

    for reg in regimes:
        reg_rows = [rr for rr in rows if rr.get("regime") == reg]
        if not reg_rows:
            continue

        # arrays
        P = np.array([[float(rr[p]) for p in param_names] for rr in reg_rows], dtype=float)

        metrics_of_interest = [
            ("hybrid_accuracy", "max"),
            ("CCR_sroa_hybrid", "max"),
            ("rad_mean", "min"),
        ]

        prcc_table: List[Dict[str, Any]] = []
        spearman_table: List[Dict[str, Any]] = []

        for metric, _dir in metrics_of_interest:
            y = np.array([float(rr[metric]) if rr[metric] not in (None, "", "None") else float("nan") for rr in reg_rows], dtype=float)
            # drop NaNs
            mask = ~np.isnan(y)
            Pm = P[mask]
            ym = y[mask]
            if len(ym) < 5:
                continue
            pr = compute_prcc(Pm, ym, param_names)
            # Spearman (rank correlation)
            yy = _rank(ym)
            for i, pname in enumerate(param_names):
                xi = _rank(Pm[:, i])
                rho = float(np.corrcoef(xi, yy)[0, 1]) if (np.std(xi) * np.std(yy)) != 0 else float("nan")
                spearman_table.append({"metric": metric, "param": pname, "spearman_rho": rho})
                prcc_table.append({"metric": metric, "param": pname, "prcc": pr[pname]})

        # Pareto front (needs float rows)
        flat_rows: List[Dict[str, Any]] = []
        for rr in reg_rows:
            flat_rows.append({
                "regime": reg,
                "sample_id": int(rr["sample_id"]),
                "seed": int(rr["seed"]),
                "lam": float(rr["lam"]),
                "gamma": float(rr["gamma"]),
                "omega": float(rr["omega"]),
                "hybrid_accuracy": float(rr["hybrid_accuracy"]),
                "CCR_sroa_hybrid": float(rr["CCR_sroa_hybrid"]) if rr["CCR_sroa_hybrid"] not in (None, "", "None") else float("nan"),
                "rad_mean": float(rr["rad_mean"]),
                "avg_defeats_total": float(rr["avg_defeats_total"]),
            })
        # Some regimes could yield nan CCR; pareto_front expects finite
        flat_rows2 = [r for r in flat_rows if not (math.isnan(r["CCR_sroa_hybrid"]) or math.isnan(r["hybrid_accuracy"]) or math.isnan(r["rad_mean"]))]
        front = pareto_front(flat_rows2, objectives=[("hybrid_accuracy", "max"), ("CCR_sroa_hybrid", "max"), ("rad_mean", "min")])

        out_reg = os.path.join(sens_dir, "analysis", reg)
        _ensure_dir(out_reg)

        # Save CSV for compatibility
        _write_csv(os.path.join(out_reg, "pareto_front.csv"), front)

        # Save XLSX (required)
        prcc_xlsx = os.path.join(out_reg, "prcc.xlsx")
        save_workbook(prcc_xlsx, {
            "prcc": {"type": "dict_rows", "rows": prcc_table},
            "spearman": {"type": "dict_rows", "rows": spearman_table},
        })
        pareto_xlsx = os.path.join(out_reg, "pareto_front.xlsx")
        save_workbook(pareto_xlsx, {"pareto_front": {"type": "dict_rows", "rows": front}})

        results["per_regime"][reg] = {
            "prcc_xlsx": prcc_xlsx,
            "pareto_xlsx": pareto_xlsx,
            "pareto_csv": os.path.join(out_reg, "pareto_front.csv"),
        }

    return results

def _regime_overrides(regime: str) -> Dict[str, Any]:
    """Map regime id to scenario overrides (kept aligned with previous prototype semantics)."""
    if regime == "R0_default":
        return {}
    if regime == "R1_expert_degraded":
        return {"expert_accuracy": 0.75}
    if regime == "R2_more_noise":
        return {"noisy_accuracy": 0.45, "noisy_attack_rate": 0.85}
    if regime == "R3_more_adversarial":
        return {"adversary_accuracy": 0.30, "adversary_attack_rate": 0.95}
    # fallback: no override
    return {}

def run_full_pipeline(
    outdir: str,
    regimes: List[str],
    seeds: List[int],
    n_samples: int,
    warmup: int,
    test: int,
    lam_min: float,
    lam_max: float,
    gamma_min: float,
    gamma_max: float,
    omega_min: float,
    omega_max: float,
    design_seed: int,
    forgetting: float,
    coalition_size: int,
    include_adversary: bool,
    drop_rate: float,
    dup_rate: float,
    trace_level: str,
    audit_max_cases: int,
    audit_policy: str,
    log_every_runs: int = 10,
    log_every_cases: int = 200,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """End-to-end pipeline: baseline compare + sensitivity + analysis + paper_outputs."""
    _ensure_dir(outdir)

    # Baseline compare (one run) for paper narrative (fast + auditable)
    baseline_dir = os.path.join(outdir, "baseline")
    _ensure_dir(baseline_dir)
    baseline = run_compare(
        outdir=os.path.join(baseline_dir, "compare_R0_default"),
        seed=seeds[0] if seeds else 7,
        warmup=warmup,
        test=test,
        lam=(lam_min + lam_max) / 2.0,
        gamma=(gamma_min + gamma_max) / 2.0,
        omega=(omega_min + omega_max) / 2.0,
        forgetting=forgetting,
        coalition_size=coalition_size,
        include_adversary=include_adversary,
        drop_rate=drop_rate,
        dup_rate=dup_rate,
        scenario_overrides=_regime_overrides("R0_default"),
        trace_level=trace_level,
        audit_max_cases=audit_max_cases,
        audit_policy=audit_policy,
        use_sqlite=False,
        log_every_cases=log_every_cases,
    )

    # Sensitivity
    sens = run_sensitivity(
        outdir=outdir,
        regimes=regimes,
        seeds=seeds,
        n_samples=n_samples,
        warmup=warmup,
        test=test,
        lam_min=lam_min,
        lam_max=lam_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        omega_min=omega_min,
        omega_max=omega_max,
        design_seed=design_seed,
        forgetting=forgetting,
        coalition_size=coalition_size,
        include_adversary=include_adversary,
        drop_rate=drop_rate,
        dup_rate=dup_rate,
        log_every_runs=log_every_runs,
        log_every_cases=0,  # keep sensitivity quiet by default
        max_workers=max_workers,
    )

    # Analysis
    analysis = analyze_sensitivity(outdir, regimes)

    # Paper outputs (consolidated)
    paper = create_paper_outputs(outdir, regimes, baseline)

    return {"baseline": baseline, "sensitivity": sens, "analysis": analysis, "paper_outputs": paper}

def create_paper_outputs(outdir: str, regimes: List[str], baseline: Dict[str, Any]) -> Dict[str, str]:
    paper_dir = os.path.join(outdir, "paper_outputs")
    _ensure_dir(paper_dir)

    # paper_metrics.xlsx: repack baseline run_summary.xlsx (stable sheets)
    bm = baseline.get("method_metrics", [])
    bc = baseline.get("case_aggregate", [])
    # audit cases are stored only when trace_level==audit; read from JSONL if exists
    audit_cases: List[Dict[str, Any]] = []
    audit_jsonl = os.path.join(baseline["outdir"], "audit_cases.jsonl")
    if os.path.exists(audit_jsonl):
        import json
        with open(audit_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    audit_cases.append(json.loads(line))
    paper_metrics = os.path.join(paper_dir, "paper_metrics.xlsx")
    save_workbook(paper_metrics, {
        "baseline_method_metrics": {"type": "dict_rows", "rows": bm},
        "baseline_case_aggregate": {"type": "dict_rows", "rows": bc},
        "baseline_audit_cases": {"type": "dict_rows", "rows": audit_cases},
    })

    # paper_sensitivity.xlsx: copy metrics.xlsx content (read CSV and write)
    sens_metrics_csv = os.path.join(outdir, "sensitivity", "metrics.csv")
    import csv
    rows: List[Dict[str, Any]] = []
    with open(sens_metrics_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    paper_sens = os.path.join(paper_dir, "paper_sensitivity.xlsx")
    save_workbook(paper_sens, {"metrics": {"type": "dict_rows", "rows": rows}})

    # paper_prcc.xlsx and paper_pareto.xlsx: merge per regime
    prcc_sheets: Dict[str, Dict[str, Any]] = {}
    pareto_sheets: Dict[str, Dict[str, Any]] = {}

    for reg in regimes:
        prcc_path = os.path.join(outdir, "sensitivity", "analysis", reg, "prcc.xlsx")
        pareto_path = os.path.join(outdir, "sensitivity", "analysis", reg, "pareto_front.xlsx")
        # We avoid reading existing xlsx; instead, re-load CSV/JSON tables from standard outputs already present.
        # Here we rebuild from CSV files we wrote.
        pareto_csv = os.path.join(outdir, "sensitivity", "analysis", reg, "pareto_front.csv")
        pareto_rows: List[Dict[str, Any]] = []
        if os.path.exists(pareto_csv):
            with open(pareto_csv, encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    pareto_rows.append(row)
        # PRCC tables: we stored inside xlsx only. Recompute quickly from metrics.csv for that regime:
        # (small overhead; improves portability)
        reg_rows = [rr for rr in rows if rr.get("regime") == reg]
        if reg_rows:
            P = np.array([[float(rr[p]) for p in ["lam","gamma","omega"]] for rr in reg_rows], dtype=float)
            metrics_of_interest = ["hybrid_accuracy", "CCR_sroa_hybrid", "rad_mean"]
            prcc_rows: List[Dict[str, Any]] = []
            for metric in metrics_of_interest:
                y = np.array([float(rr[metric]) if rr[metric] not in ("", None, "None") else float("nan") for rr in reg_rows], dtype=float)
                mask = ~np.isnan(y)
                if mask.sum() < 5:
                    continue
                pr = compute_prcc(P[mask], y[mask], ["lam","gamma","omega"])
                for pname, val in pr.items():
                    prcc_rows.append({"metric": metric, "param": pname, "prcc": val})
            prcc_sheets[f"prcc_{reg}"] = {"type": "dict_rows", "rows": prcc_rows}
        pareto_sheets[f"pareto_{reg}"] = {"type": "dict_rows", "rows": pareto_rows}

    paper_prcc = os.path.join(paper_dir, "paper_prcc.xlsx")
    save_workbook(paper_prcc, prcc_sheets or {"prcc": {"type": "kv", "items": [("note","no data") ]}})
    paper_pareto = os.path.join(paper_dir, "paper_pareto.xlsx")
    save_workbook(paper_pareto, pareto_sheets or {"pareto": {"type": "kv", "items": [("note","no data")]}})

    return {
        "paper_dir": paper_dir,
        "paper_metrics_xlsx": paper_metrics,
        "paper_sensitivity_xlsx": paper_sens,
        "paper_prcc_xlsx": paper_prcc,
        "paper_pareto_xlsx": paper_pareto,
    }
