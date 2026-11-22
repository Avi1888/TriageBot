"""Adapter from Member 1 triage records to the Member 2 simulation model."""

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from model import EmergencyDepartmentModel
from triage_rules import classify_patient


def create_triage_model(
    patient_records: Iterable[Mapping[str, Any]],
    *,
    arrival_interval: int = 1,
    **model_options: Any,
) -> EmergencyDepartmentModel:
    """Create a simulation from Member 1 records and classify on arrival.

    Member 1 records have clinical fields but no simulation arrival time. This
    adapter assigns sequential arrivals at ``arrival_interval`` minute gaps;
    explicit ``arrival_time`` values are preserved. Classification remains an
    injected callback, and any supplied ``esi_level`` takes precedence in the
    model as its documented manual-input mode.
    """
    if isinstance(arrival_interval, bool) or not isinstance(arrival_interval, int) or arrival_interval < 0:
        raise ValueError("arrival_interval must be a non-negative integer")

    scheduled_records = []
    first_clock_minute = None
    previous_clock_minute = None
    day_offset = 0
    for index, record in enumerate(patient_records):
        if not isinstance(record, Mapping):
            raise TypeError("each patient record must be a mapping")
        data = dict(record)
        arrival = data.get("arrival_time")
        if isinstance(arrival, str):
            try:
                clock = datetime.strptime(arrival, "%H:%M")
            except ValueError as exc:
                raise ValueError("string arrival_time values must use HH:MM format") from exc
            clock_minute = clock.hour * 60 + clock.minute
            if previous_clock_minute is not None and clock_minute < previous_clock_minute:
                day_offset += 24 * 60
            absolute_minute = clock_minute + day_offset
            if first_clock_minute is None:
                first_clock_minute = absolute_minute
            data["arrival_time"] = absolute_minute - first_clock_minute
            previous_clock_minute = clock_minute
        elif arrival is None:
            data["arrival_time"] = index * arrival_interval
        elif isinstance(arrival, bool) or not isinstance(arrival, int) or arrival < 0:
            raise ValueError("arrival_time must be a non-negative integer or HH:MM string")
        if "symptoms" not in data and "chief_complaint" in data:
            data["symptoms"] = data["chief_complaint"]
        scheduled_records.append(data)

    model_options.setdefault("classifier", classify_patient)
    return EmergencyDepartmentModel(patients=scheduled_records, **model_options)
