from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from experiment1_core import Direction, Experiment1Engine, Outcome, Tick


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "validated_parquet"
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_AUDIT_DIR = ROOT / "sample_audit"
FLOAT_TOL = 1e-9


class TraceEngine(Experiment1Engine):
    """Production engine plus forensic traces; event logic is unchanged."""

    def __init__(self, contract: str) -> None:
        super().__init__(contract=contract)
        self.event_snapshots: Dict[int, Dict[str, Any]] = {}
        self.transition_rows: List[Dict[str, Any]] = []

    def process_tick(self, tick: Tick) -> None:
        rs = self.state.range_state
        bullish, bearish = self._detect_transitions(tick)
        if bullish or bearish:
            self.transition_rows.append(
                {
                    "row_index": tick.row_index,
                    "timestamp_utc": tick.timestamp_utc,
                    "price": tick.price,
                    "direction": Direction.BULLISH.value if bullish else Direction.BEARISH.value,
                    "range_high_price": rs.range_high_price,
                    "range_low_price": rs.range_low_price,
                    "range_high_armed": rs.range_high_armed,
                    "range_low_armed": rs.range_low_armed,
                }
            )
        super().process_tick(tick)

    def _initialize_event(self, direction: Direction, tick: Tick) -> bool:
        created = super()._initialize_event(direction, tick)
        if created:
            event = (
                self.state.active_bullish_events[-1]
                if direction == Direction.BULLISH
                else self.state.active_bearish_events[-1]
            )
            self.event_snapshots[event.event_id] = {
                "event_row_index": event.event_row_index,
                "origin_range_high_price": event.origin_range_high_price,
                "origin_range_low_price": event.origin_range_low_price,
                "stop_loss_price": event.stop_loss_price,
                "risk_unit": event.risk_unit,
            }
        return created

    def _finalize_event(self, event: Any) -> None:
        snapshot = self.event_snapshots.setdefault(event.event_id, {})
        snapshot.update(
            {
                "termination_row_index": (
                    event.event_row_index + event.ticks_to_resolution
                    if event.ticks_to_resolution is not None
                    else None
                ),
                "internal_outcome": event.outcome,
                "internal_opposing_event_price": event.opposing_event_price,
                "internal_realized_R": event.realized_R,
            }
        )
        super()._finalize_event(event)


def equal_value(a: Any, b: Any, tol: float = FLOAT_TOL) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return str(a) == str(b)


def normalize_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def choose_samples(ledger: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    selected: List[int] = []

    groups = [
        (Direction.BULLISH.value, Outcome.CLOSED_BY_OPPOSING_EVENT.value),
        (Direction.BEARISH.value, Outcome.CLOSED_BY_OPPOSING_EVENT.value),
        (Direction.BULLISH.value, Outcome.STOP_LOSS.value),
        (Direction.BEARISH.value, Outcome.STOP_LOSS.value),
    ]

    for direction, outcome in groups:
        indices = ledger.index[
            (ledger["event_direction"] == direction) & (ledger["outcome"] == outcome)
        ].tolist()
        rng.shuffle(indices)
        selected.extend(indices[:per_group])

    resolved = ledger[ledger["realized_R"].notna()]
    if not resolved.empty:
        selected.extend(resolved.nlargest(min(3, len(resolved)), "realized_R").index.tolist())
        selected.extend(resolved.nsmallest(min(3, len(resolved)), "realized_R").index.tolist())

    timed = ledger[ledger["ticks_to_resolution"].notna()]
    if not timed.empty:
        selected.extend(timed.nlargest(min(3, len(timed)), "ticks_to_resolution").index.tolist())
        selected.extend(timed.nsmallest(min(3, len(timed)), "ticks_to_resolution").index.tolist())

    selected = list(dict.fromkeys(selected))
    return ledger.loc[selected].sort_values("event_id").copy()


def independent_lifecycle_check(
    row: pd.Series,
    snapshot: Dict[str, Any],
    prices: pd.Series,
    transition_by_row: Dict[int, str],
) -> Dict[str, Any]:
    event_row = int(snapshot["event_row_index"])
    termination_row = snapshot.get("termination_row_index")
    direction = str(row["event_direction"])
    stop = float(row["stop_loss_price"])
    expected_opposite = (
        Direction.BEARISH.value if direction == Direction.BULLISH.value else Direction.BULLISH.value
    )

    found_row: Optional[int] = None
    found_outcome: Optional[str] = None

    scan_end = int(termination_row) if termination_row is not None else len(prices) - 1
    for idx in range(event_row + 1, scan_end + 1):
        price = float(prices.iloc[idx])
        stop_hit = (
            price <= stop if direction == Direction.BULLISH.value else price >= stop
        )
        if stop_hit:
            found_row = idx
            found_outcome = Outcome.STOP_LOSS.value
            break
        if transition_by_row.get(idx) == expected_opposite:
            found_row = idx
            found_outcome = Outcome.CLOSED_BY_OPPOSING_EVENT.value
            break

    if str(row["outcome"]) == Outcome.UNRESOLVED_AT_FILE_END.value:
        return {
            "first_termination_match": found_row is None,
            "independent_first_row": found_row,
            "independent_outcome": found_outcome,
        }

    return {
        "first_termination_match": (
            found_row == int(termination_row) and found_outcome == str(row["outcome"])
        ),
        "independent_first_row": found_row,
        "independent_outcome": found_outcome,
    }


def audit_event(
    ledger_row: pd.Series,
    replay_row: pd.Series,
    snapshot: Dict[str, Any],
    prices: pd.Series,
    timestamps: pd.Series,
    transition_by_row: Dict[int, str],
) -> Dict[str, Any]:
    event_id = int(ledger_row["event_id"])
    checks: Dict[str, bool] = {}

    fields = [
        "contract",
        "event_direction",
        "entry_price",
        "origin_range_high_price",
        "origin_range_low_price",
        "range_high_before_range_low",
        "highest_price_during_lifecycle",
        "lowest_price_during_lifecycle",
        "stop_loss_price",
        "risk_unit",
        "opposing_event_price",
        "outcome",
        "realized_R",
        "seconds_to_resolution",
        "ticks_to_resolution",
    ]
    for field in fields:
        checks[f"{field}_match"] = equal_value(ledger_row.get(field), replay_row.get(field))

    checks["event_timestamp_match"] = (
        normalize_timestamp(ledger_row.get("event_timestamp_utc"))
        == normalize_timestamp(replay_row.get("event_timestamp_utc"))
    )
    checks["termination_timestamp_match"] = (
        normalize_timestamp(ledger_row.get("termination_timestamp_utc"))
        == normalize_timestamp(replay_row.get("termination_timestamp_utc"))
    )

    event_row = int(snapshot["event_row_index"])
    checks["raw_entry_price_match"] = equal_value(prices.iloc[event_row], ledger_row["entry_price"])
    checks["raw_entry_timestamp_match"] = (
        normalize_timestamp(timestamps.iloc[event_row])
        == normalize_timestamp(ledger_row["event_timestamp_utc"])
    )

    expected_risk = abs(float(ledger_row["entry_price"]) - float(ledger_row["stop_loss_price"]))
    checks["risk_formula_match"] = equal_value(expected_risk, ledger_row["risk_unit"])

    termination_row = snapshot.get("termination_row_index")
    lifecycle_end = (
        int(termination_row)
        if termination_row is not None
        else len(prices) - 1
    )
    lifecycle_prices = pd.to_numeric(
        prices.iloc[event_row : lifecycle_end + 1],
        errors="coerce",
    ).dropna()
    checks["raw_highest_price_match"] = (
        not lifecycle_prices.empty
        and equal_value(
            lifecycle_prices.max(),
            ledger_row["highest_price_during_lifecycle"],
        )
    )
    checks["raw_lowest_price_match"] = (
        not lifecycle_prices.empty
        and equal_value(
            lifecycle_prices.min(),
            ledger_row["lowest_price_during_lifecycle"],
        )
    )

    if ledger_row["outcome"] == Outcome.STOP_LOSS.value:
        checks["R_formula_match"] = equal_value(ledger_row["realized_R"], -1.0)
    elif ledger_row["outcome"] == Outcome.CLOSED_BY_OPPOSING_EVENT.value:
        entry = float(ledger_row["entry_price"])
        opposing = float(ledger_row["opposing_event_price"])
        risk = float(ledger_row["risk_unit"])
        expected_r = (
            (opposing - entry) / risk
            if ledger_row["event_direction"] == Direction.BULLISH.value
            else (entry - opposing) / risk
        )
        checks["R_formula_match"] = equal_value(ledger_row["realized_R"], expected_r)
    else:
        checks["R_formula_match"] = pd.isna(ledger_row["realized_R"])

    lifecycle = independent_lifecycle_check(
        ledger_row, snapshot, prices, transition_by_row
    )
    checks["first_termination_match"] = bool(lifecycle["first_termination_match"])

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "event_id": event_id,
        "contract": ledger_row["contract"],
        "event_direction": ledger_row["event_direction"],
        "outcome": ledger_row["outcome"],
        "event_row_index": event_row,
        "termination_row_index": snapshot.get("termination_row_index"),
        "audit_pass": not failed,
        "first_mismatch": failed[0] if failed else "",
        "failed_checks": "|".join(failed),
        "independent_first_row": lifecycle["independent_first_row"],
        "independent_outcome": lifecycle["independent_outcome"],
        **checks,
    }



def audit_dataframe(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    contract: str,
    audit_dir: Path | str,
    scan_error_count: int = 0,
    per_group: int = 5,
    seed: int = 20260722,
) -> Dict[str, Any]:
    """
    Audit the exact DataFrame used by the initial scan.

    Returns the summary consumed by run_rollover_experiment1.py.
    Reliability means the sampled replay aligned and neither scan produced
    engine errors; it is not a claim that every ledger row was independently
    reconstructed.
    """
    required_df = {"timestamp_utc", "price"}
    missing_df = required_df.difference(df.columns)
    if missing_df:
        raise ValueError(f"Audit DataFrame missing columns: {sorted(missing_df)}")

    audit_path = Path(audit_dir)
    audit_path.mkdir(parents=True, exist_ok=True)

    ledger_frame = ledger.copy()
    if not ledger_frame.empty:
        required_ledger = {
            "event_id",
            "contract",
            "event_timestamp_utc",
            "event_direction",
            "entry_price",
            "origin_range_high_price",
            "origin_range_low_price",
            "range_high_before_range_low",
            "highest_price_during_lifecycle",
            "lowest_price_during_lifecycle",
            "stop_loss_price",
            "risk_unit",
            "opposing_event_price",
            "termination_timestamp_utc",
            "outcome",
            "realized_R",
            "seconds_to_resolution",
            "ticks_to_resolution",
        }
        missing_ledger = required_ledger.difference(ledger_frame.columns)
        if missing_ledger:
            raise ValueError(
                f"Audit ledger missing columns: {sorted(missing_ledger)}"
            )

    samples = (
        choose_samples(ledger_frame, per_group, seed)
        if not ledger_frame.empty
        else ledger_frame.copy()
    )

    engine = TraceEngine(contract=contract)
    replay_rows = engine.process_dataframe(
        df.loc[:, ["timestamp_utc", "price"]].reset_index(drop=True),
        contract=contract,
    )
    replay = pd.DataFrame(replay_rows)
    if not replay.empty:
        replay = replay.set_index("event_id", drop=False)

    transition_by_row = {
        int(item["row_index"]): str(item["direction"])
        for item in engine.transition_rows
    }

    results: List[Dict[str, Any]] = []
    for _, ledger_row in samples.iterrows():
        event_id = int(ledger_row["event_id"])
        replay_missing = replay.empty or event_id not in replay.index
        snapshot_missing = event_id not in engine.event_snapshots
        if replay_missing or snapshot_missing:
            results.append(
                {
                    "event_id": event_id,
                    "contract": contract,
                    "audit_pass": False,
                    "first_mismatch": "event_missing_from_replay",
                    "failed_checks": "event_missing_from_replay",
                }
            )
            continue

        replay_row = replay.loc[event_id]
        if isinstance(replay_row, pd.DataFrame):
            replay_row = replay_row.iloc[0]

        results.append(
            audit_event(
                ledger_row=ledger_row,
                replay_row=replay_row,
                snapshot=engine.event_snapshots[event_id],
                prices=df["price"].reset_index(drop=True),
                timestamps=df["timestamp_utc"].reset_index(drop=True),
                transition_by_row=transition_by_row,
            )
        )

    results_df = pd.DataFrame(results)
    detail_path = audit_path / f"{contract}_sample_audit.csv"
    results_df.to_csv(detail_path, index=False)

    failure_counts: Dict[str, int] = {}
    if "failed_checks" in results_df.columns:
        for failed_text in results_df["failed_checks"].fillna(""):
            for name in str(failed_text).split("|"):
                if name:
                    failure_counts[name] = failure_counts.get(name, 0) + 1

    sampled = int(len(results_df))
    passed = (
        int(results_df["audit_pass"].fillna(False).sum())
        if sampled and "audit_pass" in results_df.columns
        else 0
    )
    failed = sampled - passed
    replay_error_count = len(engine.error_rows())

    # Empty ledgers align only when the independent replay is also empty.
    empty_ledger_alignment = (
        ledger_frame.empty and len(replay_rows) == 0
    )
    scan_audit_aligned = (
        empty_ledger_alignment
        if ledger_frame.empty
        else sampled > 0 and failed == 0
    )
    reliable_for_research = bool(
        scan_audit_aligned
        and replay_error_count == 0
        and int(scan_error_count) == 0
    )

    summary: Dict[str, Any] = {
        "contract": contract,
        "seed": seed,
        "per_group": per_group,
        "ledger_event_count": int(len(ledger_frame)),
        "replay_event_count": int(len(replay_rows)),
        "events_sampled": sampled,
        "events_passed": passed,
        "events_failed": failed,
        "pass_rate": (passed / sampled) if sampled else None,
        "failures_by_field": failure_counts,
        "scan_error_count": int(scan_error_count),
        "engine_errors_during_replay": replay_error_count,
        "scan_audit_aligned": scan_audit_aligned,
        "reliable_for_research": reliable_for_research,
        "reliability_scope": "sampled deterministic lifecycle audit",
        "detail_path": str(detail_path),
    }
    summary_path = audit_path / f"{contract}_sample_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary

def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, str]:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.contract:
        contract = args.contract
        parquet = data_dir / f"{contract}.parquet"
        ledger = output_dir / f"{contract}_ledger.csv"
    else:
        ledgers = sorted(output_dir.glob("*_ledger.csv"))
        ledgers = [p for p in ledgers if not p.stem.endswith("_ledger_run2")]
        if not ledgers:
            raise FileNotFoundError(f"No *_ledger.csv files found in {output_dir}")
        ledger = ledgers[0]
        contract = ledger.name[: -len("_ledger.csv")]
        parquet = data_dir / f"{contract}.parquet"

    if not parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet}")
    if not ledger.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger}")
    return parquet, ledger, contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically audit sampled Experiment 1 event lifecycles."
    )
    parser.add_argument("--contract", help="Contract stem, e.g. NQH14-CME")
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    args = parser.parse_args()

    parquet_path, ledger_path, contract = resolve_paths(args)
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    print(f"CONTRACT: {contract}", flush=True)
    print(f"READ PARQUET: {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path, columns=["timestamp_utc", "price"])
    print(f"ROWS: {len(df):,}", flush=True)

    print(f"READ LEDGER: {ledger_path}", flush=True)
    ledger = pd.read_csv(
        ledger_path,
        parse_dates=["event_timestamp_utc", "termination_timestamp_utc"],
    )
    samples = choose_samples(ledger, args.per_group, args.seed)
    print(f"SAMPLED EVENTS: {len(samples):,}", flush=True)

    print("REPLAY CONTRACT WITH FORENSIC TRACE...", flush=True)
    engine = TraceEngine(contract=contract)
    replay_rows = engine.process_dataframe(df, contract=contract)
    replay = pd.DataFrame(replay_rows).set_index("event_id", drop=False)
    transition_by_row = {
        int(item["row_index"]): str(item["direction"])
        for item in engine.transition_rows
    }

    results: List[Dict[str, Any]] = []
    for _, ledger_row in samples.iterrows():
        event_id = int(ledger_row["event_id"])
        if event_id not in replay.index or event_id not in engine.event_snapshots:
            results.append(
                {
                    "event_id": event_id,
                    "contract": contract,
                    "audit_pass": False,
                    "first_mismatch": "event_missing_from_replay",
                    "failed_checks": "event_missing_from_replay",
                }
            )
            continue
        results.append(
            audit_event(
                ledger_row=ledger_row,
                replay_row=replay.loc[event_id],
                snapshot=engine.event_snapshots[event_id],
                prices=df["price"],
                timestamps=df["timestamp_utc"],
                transition_by_row=transition_by_row,
            )
        )

    results_df = pd.DataFrame(results)
    detail_path = audit_dir / f"{contract}_sample_audit.csv"
    results_df.to_csv(detail_path, index=False)

    failure_counts: Dict[str, int] = {}
    for text in results_df.get("failed_checks", pd.Series(dtype=str)).fillna(""):
        for name in str(text).split("|"):
            if name:
                failure_counts[name] = failure_counts.get(name, 0) + 1

    passed = int(results_df["audit_pass"].fillna(False).sum()) if not results_df.empty else 0
    summary = {
        "contract": contract,
        "seed": args.seed,
        "per_group": args.per_group,
        "events_sampled": int(len(results_df)),
        "events_passed": passed,
        "events_failed": int(len(results_df) - passed),
        "pass_rate": (passed / len(results_df)) if len(results_df) else None,
        "failures_by_field": failure_counts,
        "engine_errors_during_replay": len(engine.error_rows()),
        "source_parquet": str(parquet_path),
        "source_ledger": str(ledger_path),
    }
    summary_path = audit_dir / f"{contract}_sample_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"DETAIL: {detail_path}", flush=True)
    print(f"SUMMARY: {summary_path}", flush=True)

    if summary["events_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
