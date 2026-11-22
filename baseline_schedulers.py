"""Simple specialty-constrained baselines for comparing the A* scheduler."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, Mapping

from agents import normalize_specialty, validate_esi_level
from astar_scheduler import DEFAULT_URGENCY_WEIGHTS, _normalise_inputs


def _greedy_schedule(patients: Iterable[Any], doctors: Iterable[Any], current_time: int,
                     order_key, urgency_weights: Mapping[int, float] | None = None) -> dict[str, Any]:
    started = perf_counter()
    weights = dict(DEFAULT_URGENCY_WEIGHTS if urgency_weights is None else urgency_weights)
    items, resources = _normalise_inputs(patients, doctors, current_time)
    availability = [d.available_at for d in resources]
    ordered = sorted(items, key=order_key)
    schedule = []
    unschedulable = []
    for patient in ordered:
        matches = [j for j, doctor in enumerate(resources)
                   if doctor.specialty_key == normalize_specialty(patient.required_specialty)]
        if not matches:
            unschedulable.append({"patient_id": patient.patient_id,
                                  "reason": f"No doctor covers required specialty {patient.required_specialty!r}"})
            continue
        j = min(matches, key=lambda index: (max(current_time, patient.arrival_time, availability[index]),
                                            resources[index].doctor_id))
        doctor = resources[j]
        start = max(current_time, patient.arrival_time, availability[j])
        wait = max(0, start - patient.arrival_time)
        end = start + patient.treatment_duration
        availability[j] = end
        schedule.append({
            "patient_id": patient.patient_id,
            "doctor_id": doctor.doctor_id,
            "specialty": doctor.specialty,
            "start_time": start,
            "end_time": end,
            "waiting_time": wait,
            "esi_level": patient.esi_level,
            "weighted_wait_cost": weights[patient.esi_level] * wait,
        })
    return {
        "schedule": schedule,
        "weighted_wait_cost": sum(row["weighted_wait_cost"] for row in schedule),
        "expanded_states": 0,
        "execution_time_seconds": perf_counter() - started,
        "unschedulable": unschedulable,
        "optimal": False,
        "complete": True,
        "batch_patient_ids": [p.patient_id for p in ordered if p.patient_id not in {x["patient_id"] for x in unschedulable}],
    }


def fcfs_schedule(patients: Iterable[Any], doctors: Iterable[Any], current_time: int = 0,
                  urgency_weights: Mapping[int, float] | None = None) -> dict[str, Any]:
    """First-come-first-served order with hard specialty matching."""
    return _greedy_schedule(patients, doctors, current_time,
                            lambda p: (p.arrival_time, p.patient_id), urgency_weights)


def esi_priority_schedule(patients: Iterable[Any], doctors: Iterable[Any], current_time: int = 0,
                          urgency_weights: Mapping[int, float] | None = None) -> dict[str, Any]:
    """ESI 2-first order, then arrival time and patient ID."""
    return _greedy_schedule(patients, doctors, current_time,
                            lambda p: (p.esi_level, p.arrival_time, p.patient_id), urgency_weights)


def compare_schedulers(patients: Iterable[Any], doctors: Iterable[Any], current_time: int = 0) -> dict[str, dict[str, Any]]:
    """Run A*, FCFS and ESI priority against independent copies of the same inputs."""
    patient_rows = list(patients)
    doctor_rows = list(doctors)
    from astar_scheduler import astar_schedule

    return {
        "astar": astar_schedule(patient_rows, doctor_rows, current_time),
        "fcfs": fcfs_schedule(patient_rows, doctor_rows, current_time),
        "esi_priority": esi_priority_schedule(patient_rows, doctor_rows, current_time),
    }
