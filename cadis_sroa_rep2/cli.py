from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

import argparse
import json
import os
from typing import Any, Dict, List

from .simulation import run_compare, summarize_results
from .pipeline import run_full_pipeline, run_sensitivity, analyze_sensitivity

def _json_arg(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cadis-sroa2",
        description="CADIS SROA Rep2 — Prototype v5 (Dung grounded vs Dung+Reputation hybrid).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # compare
    c = sub.add_parser("compare", help="Run one warmup+test compare and write run_summary.xlsx")
    c.add_argument("--outdir", required=True)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--warmup", type=int, default=200)
    c.add_argument("--test", type=int, default=200)
    c.add_argument("--lam", "--lambda", dest="lam", type=float, default=1.25)
    c.add_argument("--gamma", type=float, default=2.0)
    c.add_argument("--omega", type=float, default=0.8)
    c.add_argument("--forgetting", type=float, default=0.01, help="Forgetting rate f (eta_equivalent = 1-f)")
    c.add_argument("--coalition-size", type=int, default=2)
    c.add_argument("--include-adversary", action="store_true", default=True)
    c.add_argument("--no-adversary", dest="include_adversary", action="store_false")
    c.add_argument("--drop-rate", type=float, default=0.0)
    c.add_argument("--dup-rate", type=float, default=0.0)
    c.add_argument("--scenario-overrides", type=_json_arg, default={})
    # trace policy
    c.add_argument("--trace-level", choices=["none", "audit", "full"], default="audit")
    c.add_argument("--audit-max-cases", type=int, default=30)
    c.add_argument("--audit-policy", choices=["diff", "pure_fail", "hybrid_fail", "random"], default="diff")
    c.add_argument("--use-sqlite", action="store_true", default=False, help="Only meaningful with --trace-level full")
    # logging
    c.add_argument("--log-every-cases", type=int, default=200)

    # sensitivity (standalone)
    s = sub.add_parser("sensitivity", help="Run LHS sensitivity and write metrics.xlsx")
    s.add_argument("--outdir", required=True)
    s.add_argument("--regimes", nargs="+", default=["R0_default", "R1_expert_degraded", "R3_more_adversarial"])
    s.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13, 17, 19])
    s.add_argument("--n-samples", type=int, default=180)
    s.add_argument("--warmup", type=int, default=200)
    s.add_argument("--test", type=int, default=200)
    s.add_argument("--lam-min", type=float, default=1.0)
    s.add_argument("--lam-max", type=float, default=1.8)
    s.add_argument("--gamma-min", type=float, default=0.0)
    s.add_argument("--gamma-max", type=float, default=4.0)
    s.add_argument("--omega-min", type=float, default=0.0)
    s.add_argument("--omega-max", type=float, default=1.0)
    s.add_argument("--design-seed", type=int, default=123)
    s.add_argument("--forgetting", type=float, default=0.01)
    s.add_argument("--coalition-size", type=int, default=2)
    s.add_argument("--include-adversary", action="store_true", default=True)
    s.add_argument("--no-adversary", dest="include_adversary", action="store_false")
    s.add_argument("--drop-rate", type=float, default=0.0)
    s.add_argument("--dup-rate", type=float, default=0.0)
    s.add_argument("--log-every-runs", type=int, default=10)
    s.add_argument("--max-workers", type=int, default=1)

    # analyze
    a = sub.add_parser("analyze", help="Analyze sensitivity outputs and write prcc.xlsx and pareto_front.xlsx")
    a.add_argument("--outdir", required=True)
    a.add_argument("--regimes", nargs="+", default=["R0_default", "R1_expert_degraded", "R3_more_adversarial"])

    # pipeline
    pl = sub.add_parser("pipeline", help="Run baseline + sensitivity + analysis + paper_outputs")
    pl.add_argument("--outdir", required=True)
    pl.add_argument("--regimes", nargs="+", default=["R0_default", "R1_expert_degraded", "R3_more_adversarial"])
    pl.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13, 17, 19])
    pl.add_argument("--n-samples", type=int, default=180)
    pl.add_argument("--warmup", type=int, default=200)
    pl.add_argument("--test", type=int, default=200)
    pl.add_argument("--lam-min", type=float, default=1.0)
    pl.add_argument("--lam-max", type=float, default=1.8)
    pl.add_argument("--gamma-min", type=float, default=0.0)
    pl.add_argument("--gamma-max", type=float, default=4.0)
    pl.add_argument("--omega-min", type=float, default=0.0)
    pl.add_argument("--omega-max", type=float, default=1.0)
    pl.add_argument("--design-seed", type=int, default=123)
    pl.add_argument("--forgetting", type=float, default=0.01)
    pl.add_argument("--coalition-size", type=int, default=2)
    pl.add_argument("--include-adversary", action="store_true", default=True)
    pl.add_argument("--no-adversary", dest="include_adversary", action="store_false")
    pl.add_argument("--drop-rate", type=float, default=0.0)
    pl.add_argument("--dup-rate", type=float, default=0.0)
    # trace policy for baseline compare only (sensitivity is metrics-only)
    pl.add_argument("--trace-level", choices=["none", "audit", "full"], default="audit")
    pl.add_argument("--audit-max-cases", type=int, default=30)
    pl.add_argument("--audit-policy", choices=["diff", "pure_fail", "hybrid_fail", "random"], default="diff")
    pl.add_argument("--log-every-runs", type=int, default=10)
    pl.add_argument("--log-every-cases", type=int, default=200)
    pl.add_argument("--max-workers", type=int, default=1)

    # summarize
    sm = sub.add_parser("summarize", help="Summarize a cases JSONL file from --trace-level full mode")
    sm.add_argument("--jsonl", required=True)
    sm.add_argument("--out-xlsx", default=None)

    return p

def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "compare":
        run_compare(
            outdir=args.outdir,
            seed=args.seed,
            warmup=args.warmup,
            test=args.test,
            lam=args.lam,
            gamma=args.gamma,
            omega=args.omega,
            forgetting=args.forgetting,
            coalition_size=args.coalition_size,
            include_adversary=args.include_adversary,
            drop_rate=args.drop_rate,
            dup_rate=args.dup_rate,
            scenario_overrides=args.scenario_overrides,
            trace_level=args.trace_level,
            audit_max_cases=args.audit_max_cases,
            audit_policy=args.audit_policy,
            use_sqlite=args.use_sqlite,
            log_every_cases=args.log_every_cases,
        )
        print(f"Wrote: {os.path.join(args.outdir, 'run_summary.xlsx')}")
        return 0

    if args.cmd == "sensitivity":
        run_sensitivity(
            outdir=args.outdir,
            regimes=args.regimes,
            seeds=args.seeds,
            n_samples=args.n_samples,
            warmup=args.warmup,
            test=args.test,
            lam_min=args.lam_min,
            lam_max=args.lam_max,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            omega_min=args.omega_min,
            omega_max=args.omega_max,
            design_seed=args.design_seed,
            forgetting=args.forgetting,
            coalition_size=args.coalition_size,
            include_adversary=args.include_adversary,
            drop_rate=args.drop_rate,
            dup_rate=args.dup_rate,
            log_every_runs=args.log_every_runs,
            max_workers=args.max_workers,
        )
        print(f"Wrote: {os.path.join(args.outdir, 'sensitivity', 'metrics.xlsx')}")
        return 0

    if args.cmd == "analyze":
        analyze_sensitivity(args.outdir, args.regimes)
        print(f"Wrote: {os.path.join(args.outdir, 'sensitivity', 'analysis')}")
        return 0

    if args.cmd == "pipeline":
        run_full_pipeline(
            outdir=args.outdir,
            regimes=args.regimes,
            seeds=args.seeds,
            n_samples=args.n_samples,
            warmup=args.warmup,
            test=args.test,
            lam_min=args.lam_min,
            lam_max=args.lam_max,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            omega_min=args.omega_min,
            omega_max=args.omega_max,
            design_seed=args.design_seed,
            forgetting=args.forgetting,
            coalition_size=args.coalition_size,
            include_adversary=args.include_adversary,
            drop_rate=args.drop_rate,
            dup_rate=args.dup_rate,
            trace_level=args.trace_level,
            audit_max_cases=args.audit_max_cases,
            audit_policy=args.audit_policy,
            log_every_runs=args.log_every_runs,
            log_every_cases=args.log_every_cases,
            max_workers=args.max_workers,
        )
        print(f"Wrote: {os.path.join(args.outdir, 'paper_outputs')}")
        return 0

    if args.cmd == "summarize":
        out = summarize_results(args.jsonl, out_xlsx=args.out_xlsx)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    return 1
