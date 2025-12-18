from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import os
import random
from collections import Counter, defaultdict

import numpy as np

from .argumentation import AF, Argument
from .agents import SpecialistAgent
from .metrics import wilson_ci
from .sroa import SROA, SROAConfig
from .protocol import BaseStore, NullStore, JsonlCaseStore, SqliteStore
from .utils import write_json, compute_checksums, fingerprint_source, sha256_text, stable_json
from .excelio import save_workbook

HYPOTHESES = [f"H{i}" for i in range(1, 6)]

@dataclass
class Scenario:
    """Synthetic regime specification.

    The goal is not clinical realism but controlled stress-testing of orchestration
    under heterogeneous agent reliability and coalition refutations.
    """
    coalition_size: int = 2
    include_adversary: bool = True

    # reliabilities (probability of proposing ground truth)
    expert_accuracy: float = 0.90
    noisy_accuracy: float = 0.55
    adversary_accuracy: float = 0.40

    # confidence distributions (min,max)
    conf_correct_expert: Tuple[float, float] = (0.70, 0.95)
    conf_wrong_expert: Tuple[float, float] = (0.45, 0.70)
    conf_correct_noisy: Tuple[float, float] = (0.55, 0.85)
    conf_wrong_noisy: Tuple[float, float] = (0.45, 0.80)
    conf_correct_adv: Tuple[float, float] = (0.55, 0.90)
    conf_wrong_adv: Tuple[float, float] = (0.55, 0.90)

    # interaction behavior
    expert_attack_rate: float = 0.20
    noisy_attack_rate: float = 0.75
    adversary_attack_rate: float = 0.90
    target_expert_bias_noisy: float = 0.80
    target_expert_bias_adv: float = 0.95
    overconfident_rate: float = 0.50

def build_agents(scn: Scenario) -> List[SpecialistAgent]:
    agents: List[SpecialistAgent] = []
    # Expert (A1)
    agents.append(
        SpecialistAgent(
            "A1",
            accuracy=scn.expert_accuracy,
            conf_correct=scn.conf_correct_expert,
            conf_wrong=scn.conf_wrong_expert,
            attack_rate=scn.expert_attack_rate,
            target_expert_bias=0.0,
            overconfident=False,
        )
    )
    # Noisy coalition members (A2..)
    for i in range(2, 2 + scn.coalition_size):
        over = random.random() < scn.overconfident_rate
        agents.append(
            SpecialistAgent(
                f"A{i}",
                accuracy=scn.noisy_accuracy,
                conf_correct=scn.conf_correct_noisy,
                conf_wrong=scn.conf_wrong_noisy,
                attack_rate=scn.noisy_attack_rate,
                target_expert_bias=scn.target_expert_bias_noisy,
                overconfident=over,
            )
        )
    # Adversary (optional)
    if scn.include_adversary:
        agents.append(
            SpecialistAgent(
                "A_adv",
                accuracy=scn.adversary_accuracy,
                conf_correct=scn.conf_correct_adv,
                conf_wrong=scn.conf_wrong_adv,
                attack_rate=scn.adversary_attack_rate,
                target_expert_bias=scn.target_expert_bias_adv,
                overconfident=True,
            )
        )
    return agents

def _macro_f1_from_cm(cm: Dict[str, Dict[str, int]], labels: List[str]) -> float:
    # cm[true][pred] counts
    f1s: List[float] = []
    for lab in labels:
        tp = cm[lab].get(lab, 0)
        fp = sum(cm[t].get(lab, 0) for t in labels if t != lab)
        fn = sum(cm[lab].get(p, 0) for p in labels if p != lab)
        if tp == 0 and (fp + fn) == 0:
            f1 = 1.0
        elif tp == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0

def _reservoir_update(reservoir: List[Dict[str, Any]], item: Dict[str, Any], k: int, i: int, rng: random.Random) -> None:
    """Reservoir sample size k from a stream. i is 1-based index."""
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.randint(1, i)
    if j <= k:
        reservoir[j - 1] = item

def run_compare(
    outdir: str,
    seed: int,
    warmup: int,
    test: int,
    lam: float,
    gamma: float,
    omega: float,
    forgetting: float,
    coalition_size: int,
    include_adversary: bool,
    drop_rate: float = 0.0,
    dup_rate: float = 0.0,
    scenario_overrides: Optional[Dict[str, Any]] = None,
    trace_level: str = "audit",
    audit_max_cases: int = 30,
    audit_policy: str = "diff",
    use_sqlite: bool = False,
    log_every_cases: int = 200,
) -> Dict[str, Any]:
    """Run a single warmup+test experiment.

    v5 persistence:
    - Default: only aggregated outputs + paper-ready Excel.
    - Optional: audit subset of cases (JSONL) or full cases (JSONL).
    - Optional: legacy sqlite event store only if explicitly requested.
    """
    os.makedirs(outdir, exist_ok=True)

    # Deterministic seeds
    random.seed(seed)
    np.random.seed(seed)

    # Build scenario
    scn = Scenario(coalition_size=coalition_size, include_adversary=include_adversary)
    if scenario_overrides:
        for k, v in scenario_overrides.items():
            if hasattr(scn, k):
                setattr(scn, k, v)

    # Trace stores
    trace_level = (trace_level or "audit").lower()
    if trace_level not in {"none", "audit", "full"}:
        raise ValueError("trace_level must be one of: none, audit, full")

    store: BaseStore = NullStore()
    full_store: Optional[BaseStore] = None
    sqlite_store: Optional[SqliteStore] = None

    if trace_level == "audit":
        store = JsonlCaseStore(os.path.join(outdir, "audit_cases.jsonl"))
    elif trace_level == "full":
        full_store = JsonlCaseStore(os.path.join(outdir, "cases_full.jsonl"))
        if use_sqlite:
            sqlite_store = SqliteStore(os.path.join(outdir, "event_store.sqlite"))

    agent_ids = ["A1"] + [f"A{i}" for i in range(2, 2 + coalition_size)] + (["A_adv"] if include_adversary else [])
    sroa_pure = SROA(agent_ids, SROAConfig(mode="pure", lam=lam, gamma=gamma, omega=omega, forgetting=forgetting))
    sroa_hyb = SROA(agent_ids, SROAConfig(mode="hybrid", lam=lam, gamma=gamma, omega=omega, forgetting=forgetting))

    # Aggregators
    labels = HYPOTHESES
    cm: Dict[str, Dict[str, Dict[str, int]]] = {m: {t: defaultdict(int) for t in labels} for m in ["pure", "hybrid", "majority", "conf_weighted", "ds_only"]}
    correct: Dict[str, int] = {m: 0 for m in cm.keys()}
    n_test = 0

    # Paired counts pure vs hybrid
    paired = {"pure_correct": 0, "hybrid_correct": 0, "both_correct": 0, "both_wrong": 0}

    # Operational
    msg_sum = 0
    hyb_attacks_sum = 0
    hyb_defeats_sum = 0
    hyb_filtered_sum = 0
    rad_list: List[float] = []

    # Conditional metrics: expert-preservation (EPR) and coalition-correction (CCR)
    expert_total = 0
    nonexpert_total = 0
    pure_correct_expert = 0
    pure_correct_nonexpert = 0
    hyb_correct_expert = 0
    hyb_correct_nonexpert = 0

    gt_counts = Counter()
    pred_counts: Dict[str, Counter] = {m: Counter() for m in cm.keys()}

    # Audit selection
    audit_policy = (audit_policy or "diff").lower()
    if audit_policy not in {"diff", "pure_fail", "hybrid_fail", "random"}:
        raise ValueError("audit_policy must be one of: diff, pure_fail, hybrid_fail, random")
    rng_audit = random.Random(seed ^ 0xBADC0DE)
    selected: List[Dict[str, Any]] = []
    reservoir_all: List[Dict[str, Any]] = []  # for padding / random policy
    stream_i = 0

    def should_select(rec: Dict[str, Any]) -> bool:
        if audit_policy == "diff":
            return rec["pred_pure"] != rec["pred_hybrid"]
        if audit_policy == "pure_fail":
            return not rec["pure_correct"]
        if audit_policy == "hybrid_fail":
            return not rec["hybrid_correct"]
        return False

    def add_audit_candidate(rec: Dict[str, Any], reason: str) -> None:
        if len(selected) < audit_max_cases:
            rec["audit_reason"] = reason
            selected.append(rec)

    # Warmup: stabilize reputation only (no traces unless full+use_sqlite)
    for i in range(warmup):
        _run_one_case(
            case_idx=i,
            phase="warmup",
            scn=scn,
            sroa_pure=sroa_pure,
            sroa_hyb=sroa_hyb,
            drop_rate=drop_rate,
            dup_rate=dup_rate,
            sqlite_store=sqlite_store if (trace_level == "full" and use_sqlite) else None,
        )

    # Test
    for i in range(test):
        case = _run_one_case(
            case_idx=i,
            phase="test",
            scn=scn,
            sroa_pure=sroa_pure,
            sroa_hyb=sroa_hyb,
            drop_rate=drop_rate,
            dup_rate=dup_rate,
            sqlite_store=sqlite_store if (trace_level == "full" and use_sqlite) else None,
        )

        n_test += 1
        gt = case["ground_truth"]
        gt_counts[gt] += 1

        # predictions
        for mname, pred in [
            ("pure", case["pred_pure"]),
            ("hybrid", case["pred_hybrid"]),
            ("majority", case["pred_majority"]),
            ("conf_weighted", case["pred_conf_weighted"]),
            ("ds_only", case["pred_ds_only"]),
        ]:
            cm[mname][gt][pred] += 1
            pred_counts[mname][pred] += 1

        # correct
        pure_ok = case["pure_correct"]
        hyb_ok = case["hybrid_correct"]
        expert_ok = bool(case["audit_row"].get("expert_correct", False))
        if expert_ok:
            expert_total += 1
            if pure_ok: pure_correct_expert += 1
            if hyb_ok: hyb_correct_expert += 1
        else:
            nonexpert_total += 1
            if pure_ok: pure_correct_nonexpert += 1
            if hyb_ok: hyb_correct_nonexpert += 1

        if pure_ok:
            correct["pure"] += 1
        if hyb_ok:
            correct["hybrid"] += 1
        if case["majority_correct"]:
            correct["majority"] += 1
        if case["conf_weighted_correct"]:
            correct["conf_weighted"] += 1
        if case["ds_only_correct"]:
            correct["ds_only"] += 1

        # paired
        if pure_ok and hyb_ok:
            paired["both_correct"] += 1
        elif (not pure_ok) and (not hyb_ok):
            paired["both_wrong"] += 1
        else:
            if pure_ok:
                paired["pure_correct"] += 1
            if hyb_ok:
                paired["hybrid_correct"] += 1

        # operational
        msg_sum += case["message_count"]
        hyb_attacks_sum += case.get("hyb_attacks_total", 0)
        hyb_defeats_sum += case.get("hyb_defeats_total", 0)
        hyb_filtered_sum += case.get("hyb_filtered_attacks", 0)
        # rad per-case
        at = case.get("hyb_attacks_total", 0)
        fa = case.get("hyb_filtered_attacks", 0)
        rad_list.append((fa / at) if at else 0.0)

        # audit streaming
        stream_i += 1
        audit_row = case["audit_row"]
        if audit_policy == "random":
            _reservoir_update(reservoir_all, audit_row, audit_max_cases, stream_i, rng_audit)
        else:
            _reservoir_update(reservoir_all, audit_row, audit_max_cases, stream_i, rng_audit)
            if should_select(audit_row):
                add_audit_candidate(audit_row, audit_policy)

        if log_every_cases and ((i + 1) % log_every_cases == 0 or (i + 1) == test):
            print(f"[compare] seed={seed} {i+1}/{test} cases")

        # full store (case-level)
        if trace_level == "full" and full_store is not None:
            full_store.record_case(case["full_record"])

    # finalize audit selection to exact N
    if trace_level == "audit":
        if audit_policy == "random":
            selected = list(reservoir_all)[:audit_max_cases]
        else:
            if len(selected) < audit_max_cases:
                # pad with reproducible random reservoir, avoiding duplicates
                used = {(r["consultation_id"], r.get("phase", "test")) for r in selected}
                for r in reservoir_all:
                    key = (r["consultation_id"], r.get("phase", "test"))
                    if key in used:
                        continue
                    rr = dict(r)
                    rr["audit_reason"] = "fill_random"
                    selected.append(rr)
                    used.add(key)
                    if len(selected) >= audit_max_cases:
                        break
            selected = selected[:audit_max_cases]
        # persist audit JSONL (case-level)
        for r in selected:
            store.record_case(r)
        store.close()

    if full_store is not None:
        full_store.close()
    if sqlite_store is not None:
        sqlite_store.close()

    # Compute per-method metrics
    def method_summary(mname: str) -> Dict[str, Any]:
        k = correct[mname]
        n = n_test
        acc = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        mf1 = _macro_f1_from_cm(cm[mname], labels)
        return {
            "method": mname,
            "n": n,
            "accuracy": round(acc, 6),
            "wilson_lo": round(lo, 6),
            "wilson_hi": round(hi, 6),
            "macro_f1": round(mf1, 6),
        }

    method_rows = [method_summary(m) for m in ["pure", "hybrid", "majority", "conf_weighted", "ds_only"]]

    # Conditional accuracy (paper-ready):
    # EPR (Expert Preservation Rate): P(correct | expert_correct)
    # CCR (Coalition Correction Rate): P(correct | expert_wrong)
    def _safe_div(a: int, b: int) -> Optional[float]:
        return (a / b) if b else None

    pure_epr = _safe_div(pure_correct_expert, expert_total)
    pure_ccr = _safe_div(pure_correct_nonexpert, nonexpert_total)
    hyb_epr = _safe_div(hyb_correct_expert, expert_total)
    hyb_ccr = _safe_div(hyb_correct_nonexpert, nonexpert_total)

    rad_mean = float(sum(rad_list) / len(rad_list)) if rad_list else 0.0
    rad_sorted = sorted(rad_list) if rad_list else [0.0]
    def _pct(p: float) -> float:
        if not rad_sorted:
            return 0.0
        idx = int(round((len(rad_sorted)-1)*p))
        idx = 0 if idx < 0 else (len(rad_sorted)-1 if idx >= len(rad_sorted) else idx)
        return float(rad_sorted[idx])
    rad_p50 = _pct(0.50)
    rad_p95 = _pct(0.95)

    # Add paired + operational to method_metrics (on the hybrid row for convenience)
    for r in method_rows:
        if r["method"] == "pure":
            r.update({
                "EPR_sroa": None if pure_epr is None else round(float(pure_epr), 6),
                "CCR_sroa": None if pure_ccr is None else round(float(pure_ccr), 6),
            })
        if r["method"] == "hybrid":
            r.update({
                "paired_pure_correct_only": paired["pure_correct"],
                "paired_hybrid_correct_only": paired["hybrid_correct"],
                "paired_both_correct": paired["both_correct"],
                "paired_both_wrong": paired["both_wrong"],
                "EPR_sroa": None if hyb_epr is None else round(float(hyb_epr), 6),
                "CCR_sroa": None if hyb_ccr is None else round(float(hyb_ccr), 6),
                "rad_mean": round(rad_mean, 6),
                "rad_p50": round(rad_p50, 6),
                "rad_p95": round(rad_p95, 6),
                "avg_message_count": round(msg_sum / n_test, 4) if n_test else 0.0,
                "avg_attacks_total": round(hyb_attacks_sum / n_test, 4) if n_test else 0.0,
                "avg_filtered_attacks": round(hyb_filtered_sum / n_test, 4) if n_test else 0.0,
                "avg_defeats_total": round(hyb_defeats_sum / n_test, 4) if n_test else 0.0,
                "filter_ratio_mean": round((hyb_filtered_sum / hyb_attacks_sum) if hyb_attacks_sum else 0.0, 6),
                "trace_level": trace_level,
                "use_sqlite": bool(use_sqlite) if trace_level == "full" else False,
            })

    # Case aggregates
    case_agg_rows: List[Dict[str, Any]] = []
    for lab in labels:
        row = {"ground_truth": lab, "count": gt_counts.get(lab, 0)}
        for m in ["pure", "hybrid", "majority", "conf_weighted", "ds_only"]:
            row[f"pred_{m}"] = pred_counts[m].get(lab, 0)
        case_agg_rows.append(row)

    # Config (deterministic: no timestamps)
    config = {
        "author": "Omar Portilla Jaimes",
        "email": "jorge.portilla2@unipamplona.edu.co",
        "seed": seed,
        "warmup": warmup,
        "test": test,
        "lam": lam,
        "gamma": gamma,
        "omega": omega,
        "forgetting_rate": forgetting,
        "eta_equivalent": 1.0 - forgetting,
        "coalition_size": coalition_size,
        "include_adversary": include_adversary,
        "drop_rate": drop_rate,
        "dup_rate": dup_rate,
        "scenario_overrides": scenario_overrides or {},
        "trace_level": trace_level,
        "audit_policy": audit_policy,
        "audit_max_cases": audit_max_cases,
        "use_sqlite": bool(use_sqlite) if trace_level == "full" else False,
        "hypotheses": labels,
    }

    # Fingerprints
    sources = [
        os.path.join(os.path.dirname(__file__), f)
        for f in ["cli.py", "pipeline.py", "simulation.py", "sroa.py", "argumentation.py", "agents.py", "reputation.py", "metrics.py", "protocol.py", "design.py", "utils.py", "excelio.py"]
    ]
    code_fp = fingerprint_source(sources)
    cfg_fp = sha256_text(stable_json(config))

    config_out = os.path.join(outdir, "config.json")
    write_json(config_out, {**config, "code_fingerprint": code_fp, "config_fingerprint": cfg_fp})

    # Excel output
    xlsx_path = os.path.join(outdir, "run_summary.xlsx")
    sheets = {
        "config": {"type": "kv", "items": list({**config, "code_fingerprint": code_fp, "config_fingerprint": cfg_fp}.items())},
        "method_metrics": {"type": "dict_rows", "rows": method_rows},
        "case_aggregate": {"type": "dict_rows", "rows": case_agg_rows},
        "audit_cases": {"type": "dict_rows", "rows": selected if trace_level == "audit" else []},
    }
    save_workbook(xlsx_path, sheets)

    # Lightweight CSV metrics for integrations
    csv_path = os.path.join(outdir, "run_metrics.csv")
    _write_run_metrics_csv(csv_path, config, method_rows)

    # checksums (include only final artifacts; audit jsonl optional)
    artifacts = [config_out, xlsx_path, csv_path]
    if trace_level == "audit":
        aj = os.path.join(outdir, "audit_cases.jsonl")
        if os.path.exists(aj):
            artifacts.append(aj)
    if trace_level == "full":
        cj = os.path.join(outdir, "cases_full.jsonl")
        if os.path.exists(cj):
            artifacts.append(cj)
        if use_sqlite:
            db = os.path.join(outdir, "event_store.sqlite")
            if os.path.exists(db):
                artifacts.append(db)
    checks = compute_checksums(artifacts, root=outdir)
    checks_out = os.path.join(outdir, "checksums.json")
    write_json(checks_out, checks)

    return {
        "outdir": outdir,
        "config_json": config_out,
        "run_summary_xlsx": xlsx_path,
        "run_metrics_csv": csv_path,
        "checksums_json": checks_out,
        "method_metrics": method_rows,
        "case_aggregate": case_agg_rows,
    }

def _write_run_metrics_csv(path: str, config: Dict[str, Any], method_rows: List[Dict[str, Any]]) -> None:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Flatten: one row per method
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in method_rows for k in r.keys()} | {f"cfg_{k}" for k in config.keys()}))
        w.writeheader()
        for r in method_rows:
            row = {**{f"cfg_{k}": v for k, v in config.items()}, **r}
            w.writerow(row)

def _run_one_case(
    case_idx: int,
    phase: str,
    scn: Scenario,
    sroa_pure: SROA,
    sroa_hyb: SROA,
    drop_rate: float,
    dup_rate: float,
    sqlite_store: Optional[SqliteStore] = None,
) -> Dict[str, Any]:
    """Generate one synthetic case, build AF, run methods, update reputation."""
    # Logical clock as operational proxy (events suppressed in v5 default)
    clock = 0

    def tick(payload: Optional[Dict[str, Any]] = None) -> None:
        nonlocal clock
        clock += 1
        if sqlite_store is not None and payload is not None:
            sqlite_store.append_event(consultation_id=cid, logical_clock=clock, payload=payload)

    cid = f"CID-{case_idx:06d}"
    agents = build_agents(scn)
    ground_truth = random.choice(HYPOTHESES)
    tick({"type": "GROUND_TRUTH", "value": ground_truth, "phase": phase})

    # evidence events (not used by decision; kept for structure)
    for i in range(4):
        if random.random() < drop_rate:
            continue
        tick({"type": "EVIDENCE", "evidence_id": f"EV-{case_idx:06d}-{i:02d}", "hint": ground_truth})
        if random.random() < dup_rate:
            tick({"type": "EVIDENCE_DUP", "evidence_id": f"EV-{case_idx:06d}-{i:02d}", "hint": ground_truth})

    # proposals
    proposals: Dict[str, Any] = {}
    af = AF()
    agent_pred: Dict[str, str] = {}
    expert_arg_id = ""

    for ag in agents:
        prop = ag.propose(HYPOTHESES, ground_truth)
        proposals[ag.agent_id] = prop
        agent_pred[ag.agent_id] = prop.hypothesis
        arg_id = prop.arg_id
        if ag.agent_id == "A1":
            expert_arg_id = arg_id
        af.add_argument(Argument(arg_id=arg_id, agent_id=ag.agent_id, hypothesis=prop.hypothesis, confidence=prop.confidence))
        tick({"type": "ARG_PROPOSE", "agent": ag.agent_id, "arg_id": arg_id, "hyp": prop.hypothesis, "conf": prop.confidence})

    # attacks
    arg_by_agent: Dict[str, List[str]] = defaultdict(list)
    for a in af.args.values():
        arg_by_agent[a.agent_id].append(a.arg_id)

    # Each agent may attack others
    for ag in agents:
        if random.random() > ag.attack_rate:
            continue
        # Choose target argument
        if ag.target_expert_bias > 0 and random.random() < ag.target_expert_bias:
            target = expert_arg_id
        else:
            # pick a non-self argument
            candidates = [a.arg_id for a in af.args.values() if a.agent_id != ag.agent_id]
            if not candidates:
                continue
            target = random.choice(candidates)
        attacker = proposals[ag.agent_id].arg_id
        af.add_attack(attacker, target)
        tick({"type": "ARG_ATTACK", "from": attacker, "to": target})

        # occasional extra attacks (coalition reinforcement)
        if random.random() < ag.attack_rate * 0.20:
            candidates = [a.arg_id for a in af.args.values() if a.agent_id != ag.agent_id and a.arg_id != target]
            if candidates:
                t2 = random.choice(candidates)
                af.add_attack(attacker, t2)
                tick({"type": "ARG_ATTACK", "from": attacker, "to": t2})

    # baselines
    counts = Counter([a.hypothesis for a in af.args.values()])
    majority = counts.most_common(1)[0][0]
    cw = defaultdict(float)
    for a in af.args.values():
        cw[a.hypothesis] += a.confidence
    conf_weighted = max(cw.items(), key=lambda kv: kv[1])[0]
    ds_only = max(af.args.values(), key=lambda x: x.confidence).hypothesis

    # interaction graph edges (agent->agent)
    agent_ids = sorted(arg_by_agent.keys())
    idx = {aid: i for i, aid in enumerate(agent_ids)}
    edges = [(idx[af.args[u].agent_id], idx[af.args[v].agent_id]) for (u, v) in af.attacks] if af.attacks else []

    pure_pred, pure_expl = sroa_pure.decide(af, edges, idx)
    hyb_pred, hyb_expl = sroa_hyb.decide(af, edges, idx)

    # update reps
    sroa_pure.update_reputation(agent_pred, ground_truth)
    sroa_hyb.update_reputation(agent_pred, ground_truth)

    expert_h = af.args[expert_arg_id].hypothesis if expert_arg_id in af.args else None
    expert_correct = (expert_h == ground_truth) if expert_h is not None else False

    pure_correct = (pure_pred == ground_truth)
    hyb_correct = (hyb_pred == ground_truth)

    audit_row = {
        "consultation_id": cid,
        "phase": phase,
        "ground_truth": ground_truth,
        "pred_pure": pure_pred,
        "pred_hybrid": hyb_pred,
        "pure_correct": pure_correct,
        "hybrid_correct": hyb_correct,
        "expert_hypothesis": expert_h,
        "expert_correct": expert_correct,
        "accepted_args_pure": pure_expl.get("accepted_args_count", None),
        "accepted_args_hybrid": hyb_expl.get("accepted_args_count", None),
        "defeats_total": hyb_expl.get("defeats_total", None),
        "filtered_attacks": hyb_expl.get("filtered_attacks", None),
        "support_score_top3": _topk_support(hyb_expl.get("support_score", {}), k=3),
    }

    full_record = {
        "consultation_id": cid,
        "phase": phase,
        "ground_truth": ground_truth,
        "predictions": {
            "sroa_pure": pure_pred,
            "sroa_hybrid": hyb_pred,
            "majority": majority,
            "conf_weighted": conf_weighted,
            "ds_only": ds_only,
        },
        "message_count": clock,
        "explanation_pure": pure_expl,
        "explanation_hybrid": hyb_expl,
        "expert_hypothesis": expert_h,
        "expert_correct": expert_correct,
    }

    return {
        "consultation_id": cid,
        "phase": phase,
        "ground_truth": ground_truth,
        "pred_pure": pure_pred,
        "pred_hybrid": hyb_pred,
        "pred_majority": majority,
        "pred_conf_weighted": conf_weighted,
        "pred_ds_only": ds_only,
        "pure_correct": pure_correct,
        "hybrid_correct": hyb_correct,
        "majority_correct": (majority == ground_truth),
        "conf_weighted_correct": (conf_weighted == ground_truth),
        "ds_only_correct": (ds_only == ground_truth),
        "message_count": clock,
        "hyb_attacks_total": int(hyb_expl.get("attacks_total", 0) or 0),
        "hyb_defeats_total": int(hyb_expl.get("defeats_total", 0) or 0),
        "hyb_filtered_attacks": int(hyb_expl.get("filtered_attacks", 0) or 0),
        "audit_row": audit_row,
        "full_record": full_record,
    }

def _topk_support(support_score: Dict[str, Any], k: int = 3) -> str:
    if not support_score:
        return ""
    items = [(str(h), float(v)) for h, v in support_score.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:k]
    return ";".join([f"{h}:{v:.4f}" for h, v in items])


def summarize_results(path: str, out_xlsx: str | None = None) -> Dict[str, Any]:
    """Summarize a JSONL file produced in --trace-level full mode.

    This is retained for backward-compatibility and for debugging.
    For paper-ready outputs in v5, prefer run_summary.xlsx produced by `compare`/`pipeline`.
    """
    import json
    recs: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    if not recs:
        out = {"n": 0}
        if out_xlsx:
            save_workbook(out_xlsx, {"empty": {"type": "kv", "items": [("n", 0)]}})
        return out

    # expecting full_record schema
    y = [r["ground_truth"] for r in recs]
    methods = sorted(list(recs[0].get("predictions", {}).keys()))
    labels = sorted(set(y) | set([p for r in recs for p in r.get("predictions", {}).values()]))

    # compute metrics
    rows = []
    for m in methods:
        yp = [r["predictions"][m] for r in recs]
        k = sum(1 for yt, yp_ in zip(y, yp) if yt == yp_)
        n = len(y)
        acc = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        # confusion
        cm_m = {t: defaultdict(int) for t in labels}
        for yt, yp_ in zip(y, yp):
            cm_m[yt][yp_] += 1
        mf1 = _macro_f1_from_cm(cm_m, labels)
        rows.append({"method": m, "n": n, "accuracy": acc, "wilson_lo": lo, "wilson_hi": hi, "macro_f1": mf1})

    out = {"n": len(recs), "methods": methods, "metrics": rows}

    if out_xlsx:
        save_workbook(out_xlsx, {
            "summary": {"type": "dict_rows", "rows": rows},
        })
    return out


def run_compare_metrics_only(
    seed: int,
    warmup: int,
    test: int,
    lam: float,
    gamma: float,
    omega: float,
    forgetting: float,
    coalition_size: int,
    include_adversary: bool,
    drop_rate: float = 0.0,
    dup_rate: float = 0.0,
    scenario_overrides: Optional[Dict[str, Any]] = None,
    audit_max_cases: int = 0,
    audit_policy: str = "diff",
    log_every_cases: int = 0,
) -> Dict[str, Any]:
    """Fast in-memory compare for sensitivity runs (no SQLite, no JSONL, no Excel).

    Returns aggregated metrics and (optional) a small audit reservoir.
    """
    random.seed(seed)
    np.random.seed(seed)

    scn = Scenario(coalition_size=coalition_size, include_adversary=include_adversary)
    if scenario_overrides:
        for k, v in scenario_overrides.items():
            if hasattr(scn, k):
                setattr(scn, k, v)

    agent_ids = ["A1"] + [f"A{i}" for i in range(2, 2 + coalition_size)] + (["A_adv"] if include_adversary else [])
    sroa_pure = SROA(agent_ids, SROAConfig(mode="pure", lam=lam, gamma=gamma, omega=omega, forgetting=forgetting))
    sroa_hyb = SROA(agent_ids, SROAConfig(mode="hybrid", lam=lam, gamma=gamma, omega=omega, forgetting=forgetting))

    labels = HYPOTHESES
    cm: Dict[str, Dict[str, Dict[str, int]]] = {m: {t: defaultdict(int) for t in labels} for m in ["pure", "hybrid"]}
    correct = {"pure": 0, "hybrid": 0}
    paired = {"pure_correct_only": 0, "hybrid_correct_only": 0, "both_correct": 0, "both_wrong": 0}

    msg_sum = 0
    hyb_attacks_sum = 0
    hyb_defeats_sum = 0
    hyb_filtered_sum = 0
    rad_list: List[float] = []

    # Conditional metrics: EPR/CCR
    expert_total = 0
    nonexpert_total = 0
    pure_correct_expert = 0
    pure_correct_nonexpert = 0
    hyb_correct_expert = 0
    hyb_correct_nonexpert = 0

    # audit reservoir
    audit_policy = (audit_policy or "diff").lower()
    rng_audit = random.Random(seed ^ 0xA11D)
    reservoir: List[Dict[str, Any]] = []
    stream_i = 0

    for i in range(warmup):
        _run_one_case(i, "warmup", scn, sroa_pure, sroa_hyb, drop_rate, dup_rate, sqlite_store=None)

    for i in range(test):
        case = _run_one_case(i, "test", scn, sroa_pure, sroa_hyb, drop_rate, dup_rate, sqlite_store=None)
        gt = case["ground_truth"]
        p_ok = case["pure_correct"]
        h_ok = case["hybrid_correct"]

        cm["pure"][gt][case["pred_pure"]] += 1
        cm["hybrid"][gt][case["pred_hybrid"]] += 1

        if p_ok:
            correct["pure"] += 1
        if h_ok:
            correct["hybrid"] += 1

        if p_ok and h_ok:
            paired["both_correct"] += 1
        elif (not p_ok) and (not h_ok):
            paired["both_wrong"] += 1
        else:
            if p_ok:
                paired["pure_correct_only"] += 1
            if h_ok:
                paired["hybrid_correct_only"] += 1

        msg_sum += case["message_count"]
        hyb_attacks_sum += case.get("hyb_attacks_total", 0)
        hyb_defeats_sum += case.get("hyb_defeats_total", 0)
        hyb_filtered_sum += case.get("hyb_filtered_attacks", 0)
        at = case.get("hyb_attacks_total", 0)
        fa = case.get("hyb_filtered_attacks", 0)
        rad_list.append((fa / at) if at else 0.0)

        expert_ok = bool(case["audit_row"].get("expert_correct", False))
        if expert_ok:
            expert_total += 1
            if p_ok:
                pure_correct_expert += 1
            if h_ok:
                hyb_correct_expert += 1
        else:
            nonexpert_total += 1
            if p_ok:
                pure_correct_nonexpert += 1
            if h_ok:
                hyb_correct_nonexpert += 1

        # audit reservoir update
        if audit_max_cases and audit_max_cases > 0:
            stream_i += 1
            row = case["audit_row"]
            match = False
            if audit_policy == "random":
                match = True
            elif audit_policy == "diff":
                match = row["pred_pure"] != row["pred_hybrid"]
            elif audit_policy == "pure_fail":
                match = not row["pure_correct"]
            elif audit_policy == "hybrid_fail":
                match = not row["hybrid_correct"]
            if match:
                _reservoir_update(reservoir, row, audit_max_cases, stream_i, rng_audit)

        if log_every_cases and ((i + 1) % log_every_cases == 0 or (i + 1) == test):
            print(f"[sensitivity-run] seed={seed} {i+1}/{test} cases")

    n = test
    pure_acc = correct["pure"] / n if n else 0.0
    hyb_acc = correct["hybrid"] / n if n else 0.0
    pure_f1 = _macro_f1_from_cm(cm["pure"], labels)
    hyb_f1 = _macro_f1_from_cm(cm["hybrid"], labels)

    def _safe_div(a: int, b: int) -> Optional[float]:
        return (a / b) if b else None

    pure_epr = _safe_div(pure_correct_expert, expert_total)
    pure_ccr = _safe_div(pure_correct_nonexpert, nonexpert_total)
    hyb_epr = _safe_div(hyb_correct_expert, expert_total)
    hyb_ccr = _safe_div(hyb_correct_nonexpert, nonexpert_total)

    rad_mean = float(sum(rad_list) / len(rad_list)) if rad_list else 0.0
    rad_sorted = sorted(rad_list) if rad_list else [0.0]
    def _pct(p: float) -> float:
        idx = int(round((len(rad_sorted) - 1) * p))
        idx = 0 if idx < 0 else (len(rad_sorted) - 1 if idx >= len(rad_sorted) else idx)
        return float(rad_sorted[idx])
    rad_p50 = _pct(0.50)
    rad_p95 = _pct(0.95)

    return {
        "n": n,
        "pure_accuracy": pure_acc,
        "hybrid_accuracy": hyb_acc,
        "pure_macro_f1": pure_f1,
        "hybrid_macro_f1": hyb_f1,
        "paired_pure_correct_only": paired["pure_correct_only"],
        "paired_hybrid_correct_only": paired["hybrid_correct_only"],
        "paired_both_correct": paired["both_correct"],
        "paired_both_wrong": paired["both_wrong"],
        "avg_message_count": (msg_sum / n) if n else 0.0,
        "avg_attacks_total": (hyb_attacks_sum / n) if n else 0.0,
        "avg_filtered_attacks": (hyb_filtered_sum / n) if n else 0.0,
        "avg_defeats_total": (hyb_defeats_sum / n) if n else 0.0,
        "filter_ratio_mean": (hyb_filtered_sum / hyb_attacks_sum) if hyb_attacks_sum else 0.0,
        "EPR_sroa_pure": pure_epr,
        "CCR_sroa_pure": pure_ccr,
        "EPR_sroa_hybrid": hyb_epr,
        "CCR_sroa_hybrid": hyb_ccr,
        "rad_mean": rad_mean,
        "rad_p50": rad_p50,
        "rad_p95": rad_p95,
        "audit_cases": reservoir,
    }

