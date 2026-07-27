from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from audit_event_samples import audit_dataframe
from experiment1_core import run_experiment1_dataframe

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(r"D:\Market Data\validated_parquet")
ROLLOVER_MAP = Path(r"D:\Market Data\rollover\rollover_map.csv")
OUTPUT_DIR = Path(r"D:\Market Data\Experiment1")
EVENT_LEDGER_DIR = OUTPUT_DIR / "event_ledgers"
ERROR_DIR = OUTPUT_DIR / "errors"
AUDIT_DIR = OUTPUT_DIR / "audit"
MAX_CONTRACT_YEAR = 2022

for directory in (OUTPUT_DIR, EVENT_LEDGER_DIR, ERROR_DIR, AUDIT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

heartbeat_state = {"running": True, "contract": "not started", "phase": "initializing",
                   "rows": 0, "started": time.time()}


def heartbeat() -> None:
    while heartbeat_state["running"]:
        elapsed = (time.time() - heartbeat_state["started"]) / 60
        print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] HEARTBEAT | "
              f"elapsed={elapsed:.1f} min | contract={heartbeat_state['contract']} | "
              f"phase={heartbeat_state['phase']} | eligible_rows={heartbeat_state['rows']:,}", flush=True)
        time.sleep(30)


def calculate_trading_day(timestamp_series: pd.Series) -> pd.Series:
    utc = pd.to_datetime(timestamp_series, utc=True, errors="raise")
    return (utc.dt.tz_convert("America/New_York") - pd.Timedelta(hours=18)).dt.date.astype(str)


def contract_year(contract: str) -> int | None:
    match = re.search(r"NQ[HMUZ](\d{2}|\d{4})", str(contract).upper())
    if not match:
        return None
    digits = match.group(1)
    return int(digits) if len(digits) == 4 else 2000 + int(digits)


def summarize_group(group: pd.DataFrame, contract: str, trading_day: str) -> dict:
    event_count = len(group)
    closed = int((group["outcome"] == "CLOSED_BY_OPPOSING_EVENT").sum())
    stopped = int((group["outcome"] == "STOP_LOSS").sum())
    unresolved = int((group["outcome"] == "UNRESOLVED_AT_FILE_END").sum())
    resolved = pd.to_numeric(group["realized_R"], errors="coerce").dropna()
    total_r = float(resolved.sum()) if len(resolved) else 0.0
    average_r = float(resolved.mean()) if len(resolved) else None
    return {
        "contract": contract, "trading_day": trading_day, "event_count": event_count,
        "closed_by_opposing_event_count": closed, "stop_loss_count": stopped,
        "unresolved_count": unresolved,
        "closed_by_opposing_event_rate": closed / event_count if event_count else None,
        "stop_loss_rate": stopped / event_count if event_count else None,
        "unresolved_rate": unresolved / event_count if event_count else None,
        "average_R": average_r, "total_R": total_r, "expectancy_R": average_r,
    }


def main() -> None:
    if not ROLLOVER_MAP.exists():
        raise FileNotFoundError(f"Rollover map not found: {ROLLOVER_MAP}")
    rollover_map = pd.read_csv(ROLLOVER_MAP, dtype={"trading_day": "string", "active_contract": "string"})
    required = {"trading_day", "active_contract"}
    missing = required.difference(rollover_map.columns)
    if missing:
        raise ValueError(f"rollover_map.csv missing required columns: {sorted(missing)}")
    rollover_map["trading_day"] = rollover_map["trading_day"].astype(str)
    rollover_map["active_contract"] = rollover_map["active_contract"].astype(str)
    if rollover_map["trading_day"].duplicated().any():
        raise ValueError("Rollover map contains duplicate trading days")

    years = rollover_map["active_contract"].map(contract_year)
    invalid = rollover_map.loc[years.isna(), "active_contract"].drop_duplicates().tolist()
    if invalid:
        raise ValueError(f"Cannot parse contract year: {invalid[:10]}")
    rollover_map = rollover_map.loc[years <= MAX_CONTRACT_YEAR].copy()
    active_days_by_contract = {c: set(g["trading_day"]) for c, g in rollover_map.groupby("active_contract", sort=False)}
    contract_order = rollover_map["active_contract"].drop_duplicates().tolist()

    print(f"DATA DIRECTORY: {DATA_DIR}", flush=True)
    print(f"OUTPUT DIRECTORY: {OUTPUT_DIR}", flush=True)
    print(f"MAX CONTRACT YEAR: {MAX_CONTRACT_YEAR}", flush=True)
    print(f"ELIGIBLE CONTRACTS: {len(contract_order)}", flush=True)

    run_records, daily_summary_rows = [], []
    all_reliable = True
    for sequence, contract in enumerate(contract_order, 1):
        parquet_path = DATA_DIR / f"{contract}.parquet"
        heartbeat_state.update(contract=contract, phase="loading parquet", rows=0)
        if not parquet_path.exists():
            print(f"ERROR: missing {parquet_path}", flush=True)
            run_records.append({"contract": contract, "status": "MISSING_PARQUET", "reliable": False})
            all_reliable = False
            continue
        assigned_days = active_days_by_contract[contract]
        df = pd.read_parquet(parquet_path, columns=["timestamp_utc", "price"])
        source_rows = len(df)
        heartbeat_state["phase"] = "applying rollover filter"
        trading_days = calculate_trading_day(df["timestamp_utc"])
        eligible = trading_days.isin(assigned_days)
        filtered = df.loc[eligible, ["timestamp_utc", "price"]].reset_index(drop=True)
        filtered_days = trading_days.loc[eligible].reset_index(drop=True)
        heartbeat_state["rows"] = len(filtered)
        if filtered.empty:
            run_records.append({"contract": contract, "status": "NO_ELIGIBLE_TICKS", "reliable": False})
            all_reliable = False
            continue

        heartbeat_state["phase"] = "scanning events"
        started = time.time()
        ledger, summary, errors = run_experiment1_dataframe(filtered, contract)
        ledger_df = pd.DataFrame(ledger)
        ledger_path = EVENT_LEDGER_DIR / f"{contract}_event_ledger.csv"
        errors_path = ERROR_DIR / f"{contract}_errors.csv"
        ledger_df.to_csv(ledger_path, index=False)
        pd.DataFrame(errors).to_csv(errors_path, index=False)

        if not ledger_df.empty:
            event_ts = pd.to_datetime(ledger_df["event_timestamp_utc"], utc=True, errors="raise")
            ledger_df["trading_day"] = calculate_trading_day(event_ts)
            for day, group in ledger_df.groupby("trading_day", sort=True):
                daily_summary_rows.append(summarize_group(group, contract, str(day)))

        heartbeat_state["phase"] = "auditing scan"
        audit_summary = audit_dataframe(filtered, ledger_df, contract, AUDIT_DIR,
                                        scan_error_count=len(errors))
        aligned = audit_summary["scan_audit_aligned"]
        reliable = audit_summary["reliable_for_research"]
        all_reliable = all_reliable and reliable
        print("-" * 72, flush=True)
        print(f"CONTRACT: {contract}", flush=True)
        print("SCAN: COMPLETE", flush=True)
        print(f"AUDIT: {'PASS' if aligned else 'FAIL'}", flush=True)
        print(f"SCAN/AUDIT ALIGNED: {'YES' if aligned else 'NO'}", flush=True)
        print(f"RELIABLE FOR RESEARCH: {'YES' if reliable else 'NO'}", flush=True)
        print(f"SCOPE: sampled deterministic lifecycle audit ({audit_summary['events_sampled']} events)", flush=True)
        print("-" * 72, flush=True)

        run_records.append({
            "sequence_number": sequence, "contract": contract, "status": "COMPLETED",
            "source_rows": source_rows, "eligible_rows": len(filtered),
            "event_count": summary["event_count"], "scan_error_count": len(errors),
            "audit_events_sampled": audit_summary["events_sampled"],
            "scan_audit_aligned": aligned, "reliable_for_research": reliable,
            "scan_seconds": time.time() - started, "ledger_path": str(ledger_path),
        })

    heartbeat_state["phase"] = "writing aggregate artifacts"
    pd.DataFrame(daily_summary_rows).to_csv(OUTPUT_DIR / "aggregate_event_summary_by_contract_trading_day.csv", index=False)
    pd.DataFrame(run_records).to_csv(OUTPUT_DIR / "run_manifest.csv", index=False)
    manifest = {
        "run_started_utc": datetime.fromtimestamp(heartbeat_state["started"], tz=timezone.utc).isoformat(),
        "run_completed_utc": datetime.now(timezone.utc).isoformat(),
        "max_contract_year": MAX_CONTRACT_YEAR, "output_dir": str(OUTPUT_DIR),
        "all_completed_scans_reliable_for_research": all_reliable,
        "reliability_scope": "sampled deterministic lifecycle audit",
        "records": run_records,
    }
    (OUTPUT_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"FINAL RELIABILITY: {'PASS' if all_reliable else 'FAIL'}", flush=True)
    print(f"OUTPUT: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        main()
    finally:
        heartbeat_state["running"] = False
