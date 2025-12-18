import os, tempfile
from cadis_sroa_rep2.simulation import run_compare

def test_compare_outputs_exist():
    with tempfile.TemporaryDirectory() as d:
        out = run_compare(
            outdir=d,
            seed=1,
            warmup=10,
            test=20,
            lam=1.25,
            gamma=2.0,
            omega=0.8,
            forgetting=0.01,
            coalition_size=2,
            include_adversary=True,
            trace_level="audit",
            audit_max_cases=5,
            audit_policy="diff",
            log_every_cases=0,
        )
        assert os.path.exists(out["config_json"])
        assert os.path.exists(out["run_summary_xlsx"])
        assert os.path.exists(out["run_metrics_csv"])
        assert os.path.exists(out["checksums_json"])
        # audit jsonl present in audit mode
        assert os.path.exists(os.path.join(d, "audit_cases.jsonl"))
