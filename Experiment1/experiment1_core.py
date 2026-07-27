"""
Experiment 1 core deterministic event engine.

Spec contract implemented:
- Input rows are processed in authoritative input order.
- Required input fields: timestamp_utc, price.
- Per-row order:
  1) current row context
  2) rolling tick window
  2a) Update highest_price_during_lifecycle and lowest_price_during_lifecycle for every event active
  3) STOP_LOSS checks
  4) transition predicates read pre-mutation range_state
  5) opposing closures
  6) new event initialization
  7) swing confirmation
  8) range_state update
  9) persist rolling window/state for next row

No sorting. No timeout. No position logic. Ledger is authoritative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re

try:
    import pandas as pd
except Exception:  # pragma: no cover - lets pure-Python imports still work
    pd = None  # type: ignore


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class Outcome(str, Enum):
    CLOSED_BY_OPPOSING_EVENT = "CLOSED_BY_OPPOSING_EVENT"
    STOP_LOSS = "STOP_LOSS"
    UNRESOLVED_AT_FILE_END = "UNRESOLVED_AT_FILE_END"


class ErrorCode(str, Enum):
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    NULL_TIMESTAMP = "NULL_TIMESTAMP"
    NULL_PRICE = "NULL_PRICE"
    INVALID_PRICE = "INVALID_PRICE"
    EVENT_BEFORE_RANGE_READY = "EVENT_BEFORE_RANGE_READY"
    INVALID_RISK_UNIT = "INVALID_RISK_UNIT"
    INVALID_EVENT_DIRECTION = "INVALID_EVENT_DIRECTION"
    DOUBLE_TERMINATION = "DOUBLE_TERMINATION"
    UNEXPECTED_ACTIVE_STATE = "UNEXPECTED_ACTIVE_STATE"
    UNEXPECTED_EXCEPTION = "UNEXPECTED_EXCEPTION"


@dataclass(frozen=True)
class Tick:
    contract: str
    timestamp_utc: Any
    price: float
    row_index: int


@dataclass
class SwingState:
    previous_swing_high_price: Optional[float] = None
    previous_swing_high_timestamp_utc: Any = None
    previous_swing_low_price: Optional[float] = None
    previous_swing_low_timestamp_utc: Any = None


@dataclass
class RangeState:
    range_high_price: Optional[float] = None
    range_high_timestamp_utc: Any = None
    range_low_price: Optional[float] = None
    range_low_timestamp_utc: Any = None
    range_high_armed: bool = False
    range_low_armed: bool = False

    def ready(self) -> bool:
        return self.range_high_price is not None and self.range_low_price is not None


@dataclass
class EventObject:
    event_id: int
    contract: str
    event_timestamp_utc: Any
    event_direction: str
    entry_price: float
    origin_range_high_price: float
    origin_range_low_price: float
    range_high_before_range_low: bool
    highest_price_during_lifecycle: float
    lowest_price_during_lifecycle: float
    stop_loss_price: float
    risk_unit: float
    event_row_index: int
    opposing_event_price: Optional[float] = None
    termination_timestamp_utc: Any = None
    outcome: Optional[str] = None
    realized_R: Optional[float] = None
    seconds_to_resolution: Optional[float] = None
    ticks_to_resolution: Optional[int] = None
    audit_flags: str = ""
    active: bool = True

    def ledger_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("event_row_index", None)
        row.pop("active", None)
        return row


@dataclass
class ErrorRecord:
    code: str
    message: str
    contract: Optional[str] = None
    row_index: Optional[int] = None
    timestamp_utc: Any = None
    price: Any = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineState:
    swing_state: SwingState = field(default_factory=SwingState)
    range_state: RangeState = field(default_factory=RangeState)
    tick_minus_1: Optional[Tick] = None
    tick_minus_2: Optional[Tick] = None
    active_bullish_events: List[EventObject] = field(default_factory=list)
    active_bearish_events: List[EventObject] = field(default_factory=list)
    finalized_events: List[EventObject] = field(default_factory=list)
    error_records: List[ErrorRecord] = field(default_factory=list)
    next_event_id: int = 1
    rows_processed: int = 0
    events_initialized: int = 0
    events_finalized: int = 0
    errors_logged: int = 0


class Experiment1Engine:
    """Core scanner/event engine for Experiment 1."""

    REQUIRED_COLUMNS = ("timestamp_utc", "price")

    def __init__(self, contract: str = "UNKNOWN") -> None:
        self.contract = contract
        self.state = EngineState()

    def log_error(
        self,
        code: ErrorCode,
        message: str,
        tick: Optional[Tick] = None,
        **context: Any,
    ) -> None:
        self.state.error_records.append(
            ErrorRecord(
                code=code.value,
                message=message,
                contract=tick.contract if tick else self.contract,
                row_index=tick.row_index if tick else None,
                timestamp_utc=tick.timestamp_utc if tick else None,
                price=tick.price if tick else None,
                context=context,
            )
        )
        self.state.errors_logged += 1

    def process_rows(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process iterable dict rows in given order and return finalized ledger rows."""
        for row_index, row in enumerate(rows):
            try:
                tick = self._parse_row(row=row, row_index=row_index)
                if tick is None:
                    continue
                self.process_tick(tick)
            except Exception as exc:  # spec failure bucket
                self.log_error(
                    ErrorCode.UNEXPECTED_EXCEPTION,
                    str(exc),
                    context_row_index=row_index,
                )
        self.finalize_file_end()
        return self.ledger_rows()

    def process_dataframe(self, df: Any, contract: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a pandas DataFrame without sorting or pre-scanning."""
        if contract is not None:
            self.contract = contract
        self._validate_dataframe_columns(df)
        return self.process_rows(df.to_dict("records"))

    def process_tick(self, tick: Tick) -> None:
        """Apply the exact per-row order of operations from the spec."""
        state = self.state
        state.rows_processed += 1

        tick_0 = tick
        tick_minus_1 = state.tick_minus_1
        tick_minus_2 = state.tick_minus_2

        # Lifecycle extrema include every tick from initialization through termination.
        self._update_active_event_extrema(tick_0.price)

        # 3. STOP_LOSS priority before transition-event logic.
        stopped_ids = self._check_stop_losses(tick_0)

        # 4. Transition predicates read current range_state before swing/range mutation.
        bullish_transition, bearish_transition = self._detect_transitions(tick_0)

        # Defensive: both cannot be valid unless range_high < price < range_low is broken by bad state.
        if bullish_transition and bearish_transition:
            self.log_error(
                ErrorCode.UNEXPECTED_ACTIVE_STATE,
                "Both bullish and bearish transitions detected on same row.",
                tick_0,
                range_state=asdict(state.range_state),
            )
            # Stop-loss priority already ran. Do not initialize ambiguous transition.
            bullish_transition = False
            bearish_transition = False

        # 5. Opposing closure occurs after valid opposite transition detection.
        if bullish_transition:
            self._close_opposite_events(
                closing_direction=Direction.BULLISH,
                tick=tick_0,
                stopped_ids=stopped_ids,
            )
        elif bearish_transition:
            self._close_opposite_events(
                closing_direction=Direction.BEARISH,
                tick=tick_0,
                stopped_ids=stopped_ids,
            )

        # 6. Initialize new event for valid transition, then disarm triggered side.
        if bullish_transition:
            created = self._initialize_event(Direction.BULLISH, tick_0)
            if created:
                state.range_state.range_high_armed = False
        elif bearish_transition:
            created = self._initialize_event(Direction.BEARISH, tick_0)
            if created:
                state.range_state.range_low_armed = False

        # 7-8. Process every raw tick, but advance swing structure only on price change.
        price_changed = tick_minus_1 is None or tick_0.price != tick_minus_1.price

        if price_changed:
            if tick_minus_1 is not None and tick_minus_2 is not None:
                self._confirm_and_apply_swing(tick_0, tick_minus_1, tick_minus_2)

            state.tick_minus_2 = tick_minus_1

        # Keep the latest raw tick at the current price.
        state.tick_minus_1 = tick_0
        
    def finalize_file_end(self) -> None:
        """Terminate all remaining active events as UNRESOLVED_AT_FILE_END."""
        active_events = list(self.state.active_bullish_events) + list(self.state.active_bearish_events)
        last_tick = self.state.tick_minus_1
        for event in active_events:
            if not event.active:
                self.log_error(
                    ErrorCode.UNEXPECTED_ACTIVE_STATE,
                    "Inactive event found in active list at file end.",
                    last_tick,
                    event_id=event.event_id,
                )
                continue
            event.outcome = Outcome.UNRESOLVED_AT_FILE_END.value
            event.opposing_event_price = None
            event.realized_R = None
            event.termination_timestamp_utc = last_tick.timestamp_utc if last_tick else None
            event.seconds_to_resolution = self._seconds_between(event.event_timestamp_utc, event.termination_timestamp_utc)
            event.ticks_to_resolution = (last_tick.row_index - event.event_row_index) if last_tick else None
            self._finalize_event(event)
        self.state.active_bullish_events.clear()
        self.state.active_bearish_events.clear()

    def ledger_rows(self) -> List[Dict[str, Any]]:
        return [event.ledger_row() for event in self.state.finalized_events]

    def error_rows(self) -> List[Dict[str, Any]]:
        return [asdict(error) for error in self.state.error_records]

    def counters(self) -> Dict[str, int]:
        return {
            "rows_processed": self.state.rows_processed,
            "events_initialized": self.state.events_initialized,
            "events_finalized": self.state.events_finalized,
            "errors_logged": self.state.errors_logged,
        }

    def _parse_row(self, row: Dict[str, Any], row_index: int) -> Optional[Tick]:
        timestamp = row.get("timestamp_utc")
        raw_price = row.get("price")

        if timestamp is None:
            self.log_error(ErrorCode.NULL_TIMESTAMP, "timestamp_utc is NULL", None, row_index=row_index)
            return None
        if raw_price is None:
            self.log_error(ErrorCode.NULL_PRICE, "price is NULL", None, row_index=row_index, timestamp_utc=timestamp)
            return None
        try:
            price = float(raw_price)
        except Exception:
            self.log_error(
                ErrorCode.INVALID_PRICE,
                "price cannot be parsed as numeric",
                None,
                row_index=row_index,
                timestamp_utc=timestamp,
                price=raw_price,
            )
            return None
        if not isfinite(price):
            self.log_error(
                ErrorCode.INVALID_PRICE,
                "price is not finite",
                None,
                row_index=row_index,
                timestamp_utc=timestamp,
                price=raw_price,
            )
            return None
        return Tick(contract=self.contract, timestamp_utc=timestamp, price=price, row_index=row_index)

    def _validate_dataframe_columns(self, df: Any) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            for col in missing:
                self.log_error(ErrorCode.MISSING_REQUIRED_COLUMN, f"Missing required column: {col}")
            raise ValueError(f"Missing required columns: {missing}")

    def _detect_transitions(self, tick: Tick) -> Tuple[bool, bool]:
        rs = self.state.range_state
        if not rs.ready():
            return False, False
        bullish = bool(rs.range_high_armed and tick.price > rs.range_high_price)  # type: ignore[operator]
        bearish = bool(rs.range_low_armed and tick.price < rs.range_low_price)  # type: ignore[operator]
        return bullish, bearish


    def _update_active_event_extrema(self, current_price: float) -> None:
        """Update raw lifecycle extrema for all events active at row start."""
        for active_events in (
            self.state.active_bullish_events,
            self.state.active_bearish_events,
        ):
            for event in active_events:
                event.highest_price_during_lifecycle = max(
                    event.highest_price_during_lifecycle,
                    current_price,
                )
                event.lowest_price_during_lifecycle = min(
                    event.lowest_price_during_lifecycle,
                    current_price,
                )

    def _check_stop_losses(self, tick: Tick) -> set[int]:
        stopped_ids: set[int] = set()

        for event in list(self.state.active_bullish_events):
            if tick.price <= event.stop_loss_price:
                self._terminate_stop_loss(event, tick)
                self.state.active_bullish_events.remove(event)
                stopped_ids.add(event.event_id)

        for event in list(self.state.active_bearish_events):
            if tick.price >= event.stop_loss_price:
                self._terminate_stop_loss(event, tick)
                self.state.active_bearish_events.remove(event)
                stopped_ids.add(event.event_id)

        return stopped_ids

    def _terminate_stop_loss(self, event: EventObject, tick: Tick) -> None:
        if not event.active or event.outcome is not None:
            self.log_error(ErrorCode.DOUBLE_TERMINATION, "Event already finalized before STOP_LOSS", tick, event_id=event.event_id)
            return
        event.outcome = Outcome.STOP_LOSS.value
        event.opposing_event_price = None
        event.termination_timestamp_utc = tick.timestamp_utc
        event.realized_R = -1.0
        event.seconds_to_resolution = self._seconds_between(event.event_timestamp_utc, tick.timestamp_utc)
        event.ticks_to_resolution = tick.row_index - event.event_row_index
        self._finalize_event(event)

    def _close_opposite_events(self, closing_direction: Direction, tick: Tick, stopped_ids: set[int]) -> None:
        if closing_direction == Direction.BULLISH:
            opposite_list = self.state.active_bearish_events
        elif closing_direction == Direction.BEARISH:
            opposite_list = self.state.active_bullish_events
        else:
            self.log_error(ErrorCode.INVALID_EVENT_DIRECTION, "Invalid closing direction", tick, direction=closing_direction)
            return

        for event in list(opposite_list):
            if event.event_id in stopped_ids:
                continue
            if not event.active:
                self.log_error(ErrorCode.UNEXPECTED_ACTIVE_STATE, "Inactive event in active list", tick, event_id=event.event_id)
                opposite_list.remove(event)
                continue
            self._terminate_by_opposing_event(event, tick)
            opposite_list.remove(event)

    def _terminate_by_opposing_event(self, event: EventObject, tick: Tick) -> None:
        if not event.active or event.outcome is not None:
            self.log_error(ErrorCode.DOUBLE_TERMINATION, "Event already finalized before opposing closure", tick, event_id=event.event_id)
            return

        event.outcome = Outcome.CLOSED_BY_OPPOSING_EVENT.value
        event.opposing_event_price = tick.price
        event.termination_timestamp_utc = tick.timestamp_utc
        if event.event_direction == Direction.BULLISH.value:
            event.realized_R = (tick.price - event.entry_price) / event.risk_unit
        elif event.event_direction == Direction.BEARISH.value:
            event.realized_R = (event.entry_price - tick.price) / event.risk_unit
        else:
            self.log_error(ErrorCode.INVALID_EVENT_DIRECTION, "Invalid event direction on termination", tick, event_id=event.event_id)
            event.realized_R = None
        event.seconds_to_resolution = self._seconds_between(event.event_timestamp_utc, tick.timestamp_utc)
        event.ticks_to_resolution = tick.row_index - event.event_row_index
        self._finalize_event(event)

    def _initialize_event(self, direction: Direction, tick: Tick) -> bool:
        rs = self.state.range_state
        if not rs.ready():
            self.log_error(ErrorCode.EVENT_BEFORE_RANGE_READY, "Transition attempted before range ready", tick)
            return False

        origin_high = rs.range_high_price
        origin_low = rs.range_low_price
        if origin_high is None or origin_low is None:
            self.log_error(ErrorCode.EVENT_BEFORE_RANGE_READY, "Range high/low unexpectedly NULL", tick)
            return False

        if direction == Direction.BULLISH:
            stop_loss_price = origin_low
        elif direction == Direction.BEARISH:
            stop_loss_price = origin_high
        else:
            self.log_error(ErrorCode.INVALID_EVENT_DIRECTION, "Invalid event direction", tick, direction=direction)
            return False

        risk_unit = abs(tick.price - stop_loss_price)
        if risk_unit == 0 or not isfinite(risk_unit):
            self.log_error(ErrorCode.INVALID_RISK_UNIT, "risk_unit == 0 or non-finite", tick, stop_loss_price=stop_loss_price)
            return False

        event = EventObject(
            event_id=self.state.next_event_id,
            contract=tick.contract,
            event_timestamp_utc=tick.timestamp_utc,
            event_direction=direction.value,
            entry_price=tick.price,
            origin_range_high_price=origin_high,
            origin_range_low_price=origin_low,
            range_high_before_range_low=(
                rs.range_high_timestamp_utc < rs.range_low_timestamp_utc
            ),
            highest_price_during_lifecycle=tick.price,
            lowest_price_during_lifecycle=tick.price,
            stop_loss_price=stop_loss_price,
            risk_unit=risk_unit,
            event_row_index=tick.row_index,
        )
        self.state.next_event_id += 1
        self.state.events_initialized += 1
        if direction == Direction.BULLISH:
            self.state.active_bullish_events.append(event)
        else:
            self.state.active_bearish_events.append(event)
        return True

    def _finalize_event(self, event: EventObject) -> None:
        if not event.active:
            self.state.error_records.append(
                ErrorRecord(
                    code=ErrorCode.DOUBLE_TERMINATION.value,
                    message="Attempted to finalize already inactive event.",
                    contract=event.contract,
                    row_index=event.event_row_index,
                    timestamp_utc=event.event_timestamp_utc,
                    price=event.entry_price,
                    context={"event_id": event.event_id},
                )
            )
            self.state.errors_logged += 1
            return
        event.active = False
        self.state.finalized_events.append(event)
        self.state.events_finalized += 1

    def _confirm_and_apply_swing(self, tick_0: Tick, tick_minus_1: Tick, tick_minus_2: Tick) -> None:
        # Swing low: tick_0 > tick_-1 < tick_-2.
        if tick_0.price > tick_minus_1.price < tick_minus_2.price:
            self._apply_swing_low(tick_minus_1.price, tick_minus_1.timestamp_utc)

        # Swing high: tick_0 < tick_-1 > tick_-2.
        if tick_0.price < tick_minus_1.price > tick_minus_2.price:
            self._apply_swing_high(tick_minus_1.price, tick_minus_1.timestamp_utc)

    def _apply_swing_high(self, swing_high_price: float, swing_high_timestamp_utc: Any) -> None:
        ss = self.state.swing_state
        rs = self.state.range_state
        if ss.previous_swing_high_price is None:
            ss.previous_swing_high_price = swing_high_price
            ss.previous_swing_high_timestamp_utc = swing_high_timestamp_utc
        elif swing_high_price > ss.previous_swing_high_price:
            rs.range_high_price = swing_high_price
            rs.range_high_timestamp_utc = swing_high_timestamp_utc
            rs.range_high_armed = True
            ss.previous_swing_high_price = swing_high_price
            ss.previous_swing_high_timestamp_utc = swing_high_timestamp_utc
        else:
            ss.previous_swing_high_price = swing_high_price
            ss.previous_swing_high_timestamp_utc = swing_high_timestamp_utc

    def _apply_swing_low(self, swing_low_price: float, swing_low_timestamp_utc: Any) -> None:
        ss = self.state.swing_state
        rs = self.state.range_state
        if ss.previous_swing_low_price is None:
            ss.previous_swing_low_price = swing_low_price
            ss.previous_swing_low_timestamp_utc = swing_low_timestamp_utc
        elif swing_low_price < ss.previous_swing_low_price:
            rs.range_low_price = swing_low_price
            rs.range_low_timestamp_utc = swing_low_timestamp_utc
            rs.range_low_armed = True
            ss.previous_swing_low_price = swing_low_price
            ss.previous_swing_low_timestamp_utc = swing_low_timestamp_utc
        else:
            ss.previous_swing_low_price = swing_low_price
            ss.previous_swing_low_timestamp_utc = swing_low_timestamp_utc

    @staticmethod
    def _seconds_between(start: Any, end: Any) -> Optional[float]:
        if start is None or end is None:
            return None
        try:
            delta = end - start
            if hasattr(delta, "total_seconds"):
                return float(delta.total_seconds())
        except Exception:
            return None
        return None


def extract_contract_letter(contract: str) -> Optional[str]:
    """First occurrence of H, M, U, or Z after NQ in contract identifier."""
    match = re.search(r"NQ.*?([HMUZ])", str(contract))
    return match.group(1) if match else None


def utc_day(timestamp_utc: Any) -> Optional[str]:
    if timestamp_utc is None:
        return None
    try:
        return str(timestamp_utc.date())
    except Exception:
        text = str(timestamp_utc)
        return text[:10] if len(text) >= 10 else text


def session_6h_utc(timestamp_utc: Any) -> Optional[str]:
    """
    UTC 6-hour session bucket fallback. The spec requests recurrence per session but does
    not define session boundaries in this file, so this returns deterministic UTC buckets.
    Replace with EST session classifier if/when session spec is provided.
    """
    if timestamp_utc is None:
        return None
    try:
        hour = int(timestamp_utc.hour)
    except Exception:
        return None
    start = (hour // 6) * 6
    end = (start + 6) % 24
    return f"UTC_{start:02d}_{end:02d}"


def aggregate_summary(ledger_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Minimal aggregate summary matching the spec fields."""
    event_count = len(ledger_rows)
    closed = [r for r in ledger_rows if r.get("outcome") == Outcome.CLOSED_BY_OPPOSING_EVENT.value]
    stopped = [r for r in ledger_rows if r.get("outcome") == Outcome.STOP_LOSS.value]
    unresolved = [r for r in ledger_rows if r.get("outcome") == Outcome.UNRESOLVED_AT_FILE_END.value]
    resolved_r = [float(r["realized_R"]) for r in ledger_rows if r.get("realized_R") is not None]

    def rate(count: int) -> Optional[float]:
        return count / event_count if event_count else None

    total_r = sum(resolved_r) if resolved_r else 0.0
    average_r = total_r / len(resolved_r) if resolved_r else None

    recurrence_by_contract_letter: Dict[str, int] = {}
    recurrence_by_day: Dict[str, int] = {}
    recurrence_by_session: Dict[str, int] = {}

    for row in ledger_rows:
        letter = extract_contract_letter(str(row.get("contract", "")))
        if letter:
            recurrence_by_contract_letter[letter] = recurrence_by_contract_letter.get(letter, 0) + 1
        day = utc_day(row.get("event_timestamp_utc"))
        if day:
            recurrence_by_day[day] = recurrence_by_day.get(day, 0) + 1
        session = session_6h_utc(row.get("event_timestamp_utc"))
        if session:
            recurrence_by_session[session] = recurrence_by_session.get(session, 0) + 1

    return {
        "event_count": event_count,
        "closed_by_opposing_event_count": len(closed),
        "stop_loss_count": len(stopped),
        "unresolved_count": len(unresolved),
        "closed_by_opposing_event_rate": rate(len(closed)),
        "stop_loss_rate": rate(len(stopped)),
        "unresolved_rate": rate(len(unresolved)),
        "average_R": average_r,
        "total_R": total_r,
        "expectancy_R": average_r,
        "total_recurrence_per_contract_letter": recurrence_by_contract_letter,
        "total_recurrence_per_day": recurrence_by_day,
        "total_recurrence_per_session": recurrence_by_session,
    }


def run_experiment1_dataframe(df: Any, contract: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Convenience wrapper: returns ledger_rows, aggregate_summary, error_rows."""
    engine = Experiment1Engine(contract=contract)
    ledger = engine.process_dataframe(df, contract=contract)
    return ledger, aggregate_summary(ledger), engine.error_rows()


__all__ = [
    "Direction",
    "Outcome",
    "ErrorCode",
    "Tick",
    "SwingState",
    "RangeState",
    "EventObject",
    "ErrorRecord",
    "EngineState",
    "Experiment1Engine",
    "aggregate_summary",
    "run_experiment1_dataframe",
]
