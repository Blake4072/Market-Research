from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path(r"D:\Market Data\Experiment1")
LEDGER_DIR = OUTPUT_DIR / "event_ledgers"
OUT_PATH = OUTPUT_DIR / "aggregate_event_summary_by_contract_trading_day.csv"


def trading_day(series: pd.Series) -> pd.Series:
    utc = pd.to_datetime(series, utc=True, errors="raise")
    return (utc.dt.tz_convert("America/New_York") - pd.Timedelta(hours=18)).dt.date.astype(str)


def summarize(group: pd.DataFrame, contract: str, day: str) -> dict:
    n = len(group)
    closed = int((group["outcome"] == "CLOSED_BY_OPPOSING_EVENT").sum())
    stopped = int((group["outcome"] == "STOP_LOSS").sum())
    unresolved = int((group["outcome"] == "UNRESOLVED_AT_FILE_END").sum())
    r = pd.to_numeric(group["realized_R"], errors="coerce").dropna()
    total_r = float(r.sum()) if len(r) else 0.0
    average_r = float(r.mean()) if len(r) else None
    return {"contract": contract, "trading_day": day, "event_count": n,
            "closed_by_opposing_event_count": closed, "stop_loss_count": stopped,
            "unresolved_count": unresolved,
            "closed_by_opposing_event_rate": closed / n if n else None,
            "stop_loss_rate": stopped / n if n else None,
            "unresolved_rate": unresolved / n if n else None,
            "average_R": average_r, "total_R": total_r, "expectancy_R": average_r}


ledger_files = sorted(LEDGER_DIR.glob("*_event_ledger.csv"))
if not ledger_files:
    raise SystemExit(f"No individual event ledgers found in {LEDGER_DIR}")
rows = []
for path in ledger_files:
    ledger = pd.read_csv(path)
    if ledger.empty:
        continue
    ledger["trading_day"] = trading_day(ledger["event_timestamp_utc"])
    contract = str(ledger["contract"].iloc[0])
    for day, group in ledger.groupby("trading_day", sort=True):
        rows.append(summarize(group, contract, str(day)))
pd.DataFrame(rows).sort_values(["contract", "trading_day"]).to_csv(OUT_PATH, index=False)
print(f"Wrote: {OUT_PATH}")
