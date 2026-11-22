"""Exact A* scheduler for small specialty-constrained ED batches.

The search minimizes experimental urgency-weighted waiting. It does not classify
patients and it never changes a treatment already in progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
import math
from time import perf_counter
from typing import Any, Iterable, Mapping

from agents import normalize_specialty, validate_esi_level

DEFAULT_URGENCY_WEIGHTS = {2: 4, 3: 3, 4: 2, 5: 1}


@dataclass(frozen=True)
class _Patient:
    patient_id: str
    arrival_time: int
    esi_level: int
    required_specialty: str
    treatment_duration: int


@dataclass(frozen=True)
class _Doctor:
    doctor_id: str
    specialty_key: str
    specialty: str
    available_at: int


def _value(record: Any, name: str, default: Any = None) -> Any:
    return record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)


def _normalise_inputs(patients: Iterable[Any], doctors: Iterable[Any], current_time: int):
    normalized_patients = []
    seen_ids = set()
    for patient in patients:
        patient_id = _value(patient, "patient_id")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("Each patient needs a non-empty patient_id")
        if patient_id in seen_ids:
            raise ValueError(f"Duplicate patient_id: {patient_id}")
        seen_ids.add(patient_id)
        level = validate_esi_level(_value(patient, "esi_level"))
        if level == 1:
            raise ValueError("ESI 1 patients must use the separate immediate-emergency pathway")
        arrival = _value(patient, "arrival_time", current_time)
        duration = _value(patient, "treatment_duration", 1)
        if isinstance(arrival, bool) or not isinstance(arrival, int) or arrival < 0:
            raise ValueError("arrival_time must be a non-negative integer")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
            raise ValueError("treatment_duration must be a positive integer")
        specialty = _value(patient, "required_specialty", "General Physician")
        normalized_patients.append(_Patient(patient_id, arrival, level, specialty, duration))

    normalized_doctors = []
    for doctor in doctors:
        if isinstance(doctor, tuple) and len(doctor) == 2:
            doctor_id, specialty = doctor
            available_at = current_time
        else:
            doctor_id = _value(doctor, "doctor_id")
            specialty = _value(doctor, "specialty")
            available_at = _value(doctor, "available_at", current_time)
        if not isinstance(doctor_id, str) or not doctor_id:
            raise ValueError("Each doctor needs a non-empty doctor_id")
        if isinstance(available_at, bool) or not isinstance(available_at, int) or available_at < 0:
            raise ValueError("doctor available_at must be a non-negative integer")
        normalized_doctors.append(_Doctor(doctor_id, normalize_specialty(specialty), specialty,
                                          max(current_time, available_at)))
    normalized_patients.sort(key=lambda p: (p.arrival_time, p.esi_level, p.patient_id))
    normalized_doctors.sort(key=lambda d: d.doctor_id)
    return normalized_patients, normalized_doctors


class AStarScheduler:
    """Find a minimum urgency-weighted-wait schedule for a finite patient batch.

    Each action chooses a remaining patient and an eligible doctor. The state
    key is (remaining patient indexes, doctor next-available times); partial
    assignments are retained on the queue node to reconstruct the winning plan.
    """

    def __init__(self, urgency_weights: Mapping[int, float] | None = None,
                 max_expansions: int | None = 250_000) -> None:
        self.urgency_weights = dict(DEFAULT_URGENCY_WEIGHTS if urgency_weights is None else urgency_weights)
        if max_expansions is not None and (isinstance(max_expansions, bool) or
                                           not isinstance(max_expansions, int) or max_expansions < 1):
            raise ValueError("max_expansions must be a positive integer or None")
        if set(self.urgency_weights) != {2, 3, 4, 5} or any(
            isinstance(weight, bool) or not isinstance(weight, (int, float)) or
            not math.isfinite(weight) or weight < 0
            for weight in self.urgency_weights.values()
        ):
            raise ValueError("urgency_weights must contain non-negative weights for ESI 2-5")
        self.max_expansions = max_expansions

    def schedule(self, patients: Iterable[Any], doctors: Iterable[Any], current_time: int = 0,
                 use_heuristic: bool = True) -> dict[str, Any]:
        """Return the best schedule and search diagnostics for one batch."""
        if isinstance(current_time, bool) or not isinstance(current_time, int) or current_time < 0:
            raise ValueError("current_time must be a non-negative integer")
        started = perf_counter()
        items, resources = _normalise_inputs(patients, doctors, current_time)
        by_id = {p.patient_id: p for p in items}
        eligible_ids = {d.specialty_key for d in resources}
        unschedulable = [
            {"patient_id": p.patient_id,
             "reason": f"No doctor covers required specialty {p.required_specialty!r}"}
            for p in items if normalize_specialty(p.required_specialty) not in eligible_ids
        ]
        feasible = [p for p in items if normalize_specialty(p.required_specialty) in eligible_ids]
        index_by_id = {p.patient_id: i for i, p in enumerate(feasible)}
        eligible_doctors = {
            i: tuple(j for j, d in enumerate(resources) if d.specialty_key == normalize_specialty(p.required_specialty))
            for i, p in enumerate(feasible)
        }
        initial_availability = tuple(d.available_at for d in resources)
        remaining = tuple(range(len(feasible)))
        start_key = (remaining, initial_availability)

        def heuristic(left: tuple[int, ...], availability: tuple[int, ...]) -> float:
            if not use_heuristic:
                return 0.0
            estimate = 0.0
            for i in left:
                p = feasible[i]
                earliest = min(max(current_time, p.arrival_time, availability[j]) for j in eligible_doctors[i])
                estimate += self.urgency_weights[p.esi_level] * max(0, earliest - p.arrival_time)
            return estimate

        serial = count()
        # Queue entries include the full partial plan, which is excluded from the canonical key.
        open_heap = [(heuristic(remaining, initial_availability), 0.0, next(serial), start_key, ())]
        best_g = {start_key: 0.0}
        expanded = 0
        best_goal = None
        optimal = True

        while open_heap:
            f_score, g_score, _, key, partial = heappop(open_heap)
            if g_score != best_g.get(key):
                continue
            remaining_ids, availability = key
            if not remaining_ids:
                best_goal = (g_score, partial)
                break
            if self.max_expansions is not None and expanded >= self.max_expansions:
                optimal = False
                break
            expanded += 1

            # Stable action order makes equal-cost schedules reproducible.
            for i in remaining_ids:
                patient = feasible[i]
                remaining_next = tuple(index for index in remaining_ids if index != i)
                seen_equivalent = set()
                for j in eligible_doctors[i]:
                    doctor = resources[j]
                    # Same-specialty doctors at the same availability produce an
                    # identical search state; keep the lexically first assignment.
                    equivalent = (doctor.specialty_key, availability[j])
                    if equivalent in seen_equivalent:
                        continue
                    seen_equivalent.add(equivalent)
                    start = max(current_time, patient.arrival_time, availability[j])
                    wait = max(0, start - patient.arrival_time)
                    end = start + patient.treatment_duration
                    availability_next = list(availability)
                    availability_next[j] = end
                    availability_next = tuple(availability_next)
                    next_key = (remaining_next, availability_next)
                    next_g = g_score + self.urgency_weights[patient.esi_level] * wait
                    if next_g >= best_g.get(next_key, float("inf")):
                        continue
                    best_g[next_key] = next_g
                    item = {
                        "patient_id": patient.patient_id,
                        "doctor_id": doctor.doctor_id,
                        "specialty": doctor.specialty,
                        "start_time": start,
                        "end_time": end,
                        "waiting_time": wait,
                        "esi_level": patient.esi_level,
                        "weighted_wait_cost": self.urgency_weights[patient.esi_level] * wait,
                    }
                    next_partial = partial + (item,)
                    priority = next_g + heuristic(remaining_next, availability_next)
                    heappush(open_heap, (priority, next_g, next(serial), next_key, next_partial))

        if best_goal is not None:
            weighted_cost, schedule = best_goal
            schedule = list(schedule)
        else:
            weighted_cost, schedule = None, []
            optimal = False
        return {
            "schedule": schedule,
            "weighted_wait_cost": weighted_cost,
            "expanded_states": expanded,
            "execution_time_seconds": perf_counter() - started,
            "unschedulable": unschedulable,
            "optimal": optimal and best_goal is not None,
            "complete": best_goal is not None,
            "batch_patient_ids": [p.patient_id for p in feasible],
        }

    def estimate_remaining_cost(self, patients: Iterable[Any], doctors: Iterable[Any],
                                remaining_patient_ids: Iterable[str],
                                doctor_available_at: Mapping[str, int], current_time: int = 0) -> float:
        """Public form of h(n): independent earliest eligible start per patient."""
        items, resources = _normalise_inputs(patients, doctors, current_time)
        by_id = {p.patient_id: p for p in items}
        estimate = 0.0
        for patient_id in remaining_patient_ids:
            p = by_id[patient_id]
            matched = [d for d in resources if d.specialty_key == normalize_specialty(p.required_specialty)]
            if not matched:
                raise ValueError(f"No doctor covers required specialty {p.required_specialty!r}")
            earliest = min(max(current_time, p.arrival_time, doctor_available_at.get(d.doctor_id, d.available_at))
                           for d in matched)
            estimate += self.urgency_weights[p.esi_level] * max(0, earliest - p.arrival_time)
        return estimate


def astar_schedule(patients: Iterable[Any], doctors: Iterable[Any], current_time: int = 0,
                   urgency_weights: Mapping[int, float] | None = None,
                   max_expansions: int | None = 250_000,
                   use_heuristic: bool = True) -> dict[str, Any]:
    """Convenience wrapper around :class:`AStarScheduler`."""
    return AStarScheduler(urgency_weights, max_expansions).schedule(
        patients, doctors, current_time, use_heuristic
    )


def brute_force_optimal_cost(patients: Iterable[Any], doctors: Iterable[Any], current_time: int = 0,
                             urgency_weights: Mapping[int, float] | None = None) -> float | None:
    """Exhaustive small-instance oracle used for validating A* and its heuristic."""
    weights = dict(DEFAULT_URGENCY_WEIGHTS if urgency_weights is None else urgency_weights)
    items, resources = _normalise_inputs(patients, doctors, current_time)
    if any(not any(d.specialty_key == normalize_specialty(p.required_specialty) for d in resources)
           for p in items):
        return None

    def visit(remaining: tuple[int, ...], availability: tuple[int, ...], cost: float) -> float:
        if not remaining:
            return cost
        best = float("inf")
        for i in remaining:
            patient = items[i]
            left = tuple(index for index in remaining if index != i)
            for j, doctor in enumerate(resources):
                if doctor.specialty_key != normalize_specialty(patient.required_specialty):
                    continue
                start = max(current_time, patient.arrival_time, availability[j])
                wait_cost = weights[patient.esi_level] * max(0, start - patient.arrival_time)
                next_availability = list(availability)
                next_availability[j] = start + patient.treatment_duration
                best = min(best, visit(left, tuple(next_availability), cost + wait_cost))
        return best

    return visit(tuple(range(len(items))), tuple(d.available_at for d in resources), 0.0)
