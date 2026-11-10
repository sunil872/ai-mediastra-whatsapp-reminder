"""RefillCare ML lifecycle CLI. Suitable for cron / Windows Task Scheduler.

Examples:
    python scripts/ml_cli.py plan                                     # train/validation/test windows from current data
    python scripts/ml_cli.py retrain                                  # train -> select on validation -> gate on test
    python scripts/ml_cli.py retrain --promote                        # ... and promote the best if the gate approves
    python scripts/ml_cli.py freshness                                # does fresher training data improve accuracy?
    python scripts/ml_cli.py rules-ab --version <id>                  # which prediction rules help? (read-only)
    python scripts/ml_cli.py rolling --folds 3                        # is the gain real across several unseen months?
    python scripts/ml_cli.py list
    python scripts/ml_cli.py train                                    # cutoff defaults to the planned train window
    python scripts/ml_cli.py evaluate --versions legacy-human-hgb baseline-median-v1 <new_version> --candidate <new_version>
    python scripts/ml_cli.py gate --version <version_id>
    python scripts/ml_cli.py promote --version <version_id>          # only succeeds if the gate approves
    python scripts/ml_cli.py outcomes                                 # refresh live outcomes + print summary
    python scripts/ml_cli.py expire                                   # expire unsent past-due reminder stages
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import SessionLocal  # noqa: E402
from refillcare.ml import registry as reg  # noqa: E402


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("plan")
    fr = sub.add_parser("freshness")
    fr.add_argument("--step-days", type=int, help="gap between the two training cutoffs (default 30)")
    fr.add_argument("--max-events", type=int, default=None, help="cap events per evaluation (default: all, or ML_MAX_EVAL_EVENTS)")
    fr.add_argument("--max-train-events", type=int, default=150000)
    ab = sub.add_parser("rules-ab", help="compare every prediction rule on the same unseen events (read-only)")
    ab.add_argument("--version", help="model version (default: production)")
    ab.add_argument("--max-events", type=int, default=None, help="cap events per evaluation (default: all, or ML_MAX_EVAL_EVENTS)")
    ro = sub.add_parser("rolling", help="score the model + current rules on several back-to-back unseen months")
    ro.add_argument("--folds", type=int, help="number of test months (default ML_ROLLING_FOLDS or 3)")
    ro.add_argument("--max-events", type=int, default=None, help="cap events per evaluation (default: all, or ML_MAX_EVAL_EVENTS)")
    ro.add_argument("--max-train-events", type=int, default=150000)
    rt = sub.add_parser("retrain")
    rt.add_argument("--promote", action="store_true", help="promote the validation-selected model if the test gate approves")
    rt.add_argument("--by", default="cli")
    rt.add_argument("--max-events", type=int, default=None, help="cap events per evaluation (default: all, or ML_MAX_EVAL_EVENTS)")
    e = sub.add_parser("evaluate")
    e.add_argument("--versions", nargs="*")
    e.add_argument("--candidate")
    e.add_argument("--start", type=date.fromisoformat)
    e.add_argument("--end", type=date.fromisoformat)
    e.add_argument("--max-events", type=int, default=None, help="cap events per evaluation (default: all, or ML_MAX_EVAL_EVENTS)")
    t = sub.add_parser("train")
    t.add_argument("--train-end", type=date.fromisoformat, help="default: end of the planned train window")
    t.add_argument("--train-start", type=date.fromisoformat)
    t.add_argument("--max-events", type=int, default=150000)
    g = sub.add_parser("gate")
    g.add_argument("--version", required=True)
    g.add_argument("--run")
    pr = sub.add_parser("promote")
    pr.add_argument("--version", required=True)
    pr.add_argument("--run")
    pr.add_argument("--by", default="cli")
    sub.add_parser("outcomes")
    sub.add_parser("expire")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        if args.cmd == "list":
            reg.ensure_bootstrap_versions(db)
            _print([reg.to_dict(v, db) for v in reg.list_versions(db)])
        elif args.cmd == "plan":
            from refillcare.ml.service import data_freshness, load_history

            _print(data_freshness(db, load_history()))
        elif args.cmd == "freshness":
            from refillcare.ml.service import freshness_experiment

            _print(freshness_experiment(db, step_days=args.step_days, created_by="cli",
                                        max_eval_events=args.max_events, max_train_events=args.max_train_events))
        elif args.cmd == "rules-ab":
            from refillcare.ml.service import compare_rules

            _print(compare_rules(db, version_id=args.version, max_events=args.max_events))
        elif args.cmd == "rolling":
            from refillcare.ml.service import rolling_evaluation

            _print(rolling_evaluation(db, n_folds=args.folds, created_by="cli",
                                      max_eval_events=args.max_events, max_train_events=args.max_train_events))
        elif args.cmd == "retrain":
            from refillcare.ml.service import retrain_cycle

            res = retrain_cycle(db, promote_best=args.promote, created_by=args.by, max_eval_events=args.max_events)
            _print(res)
            return 0 if (res["promoted"] or not args.promote) else 2
        elif args.cmd == "evaluate":
            from refillcare.ml.service import run_evaluation

            run = run_evaluation(db, version_ids=args.versions, candidate_version_id=args.candidate,
                                 eval_start=args.start, eval_end=args.end, max_events=args.max_events, created_by="cli")
            r = run.report or {}
            _print({"run_id": run.run_id, "status": run.status, "error": run.error,
                    "window": r.get("evaluation_window"), "event_counts": r.get("event_counts"),
                    "paired_methods": (r.get("paired_comparison") or {}).get("methods"),
                    "diffs": (r.get("paired_comparison") or {}).get("diffs"), "notes": r.get("notes")})
        elif args.cmd == "train":
            from refillcare.ml.service import load_history, plan_time_splits
            from refillcare.ml.training import train_interval_model

            hist = load_history()
            train_end = args.train_end or plan_time_splits(hist)["train"]["end"]
            _print(train_interval_model(db, hist, train_end=train_end, train_start=args.train_start,
                                        max_events=args.max_events, created_by="cli"))
        elif args.cmd == "gate":
            from refillcare.ml.service import gate_for_version

            _print(gate_for_version(db, args.version, args.run))
        elif args.cmd == "promote":
            from refillcare.ml.service import promote

            res = promote(db, args.version, promoted_by=args.by, evaluation_run_id=args.run)
            _print(res)
            return 0 if res.get("promoted") else 2
        elif args.cmd == "outcomes":
            from refillcare.ml.outcomes import outcome_summary, refresh_outcomes
            from refillcare.ml.service import load_history

            _print({"refresh": refresh_outcomes(db, load_history()), "summary": outcome_summary(db, window_days=90)})
        elif args.cmd == "expire":
            from api.services import EnterpriseServices
            from refillcare.engine.persistence import RefillPersistenceManager

            today = EnterpriseServices(db).forecast_as_of()
            _print({"expired": RefillPersistenceManager().expire_past_due_stages(db, today=today), "as_of": today})
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
