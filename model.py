"""Minute-based Mesa emergency-department model with specialty-aware A* plans."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from mesa import Model

from agents import DoctorAgent, PatientAgent, SPECIALTIES, normalize_specialty, validate_esi_level
from astar_scheduler import AStarScheduler, DEFAULT_URGENCY_WEIGHTS
from metrics import SimulationMetrics

PatientRecord = Mapping[str, Any]
Classifier = Callable[[PatientRecord], Mapping[str, Any]]
DEFAULT_DOCTOR_ROSTER = (
    ("D001", "General Physician"), ("D002", "General Physician"),
    ("D003", "Surgeon"), ("D004", "Cardiologist"), ("D005", "Neurologist"),
)
LIMITED_DOCTOR_ROSTER = DEFAULT_DOCTOR_ROSTER[:1]
SCENARIOS = ("normal", "surge", "limited_staffing")


class EmergencyDepartmentModel(Model):
    """Simulate arrivals, A* scheduling, treatment, and performance.

    A future Member 1 classifier can be supplied as ``classifier=classify_patient``;
    it must return a mapping containing ``level`` and may include ``reason`` and
    ``trace``. No ESI decision logic lives in this module.

    ESI 1 cases enter a dedicated synthetic emergency-treatment pathway and do
    not enter the ordinary scheduler or consume a routine doctor slot. This is
    an educational modeling assumption, not a clinical workflow recommendation.
    """

    def __init__(
        self,
        patients: Iterable[PatientRecord] | None = None,
        doctor_roster: Iterable[Mapping[str, str] | tuple[str, str]] | None = None,
        n_doctors: int | None = None,
        arrival_schedule: Mapping[int, Iterable[PatientRecord]] | None = None,
        arrival_rate: float | None = None,
        treatment_duration: int | tuple[int, int] = (10, 30),
        duration: int = 120,
        scenario: str = "normal",
        classifier: Classifier | None = None,
        urgency_weights: Mapping[int, float] | None = None,
        batch_size: int = 8,
        seed: int | None = None,
        max_astar_expansions: int | None = 250_000,
    ) -> None:
        super().__init__(rng=seed)
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
            raise ValueError("duration must be a non-negative integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if arrival_rate is not None and (not isinstance(arrival_rate, (int, float)) or not math.isfinite(arrival_rate) or arrival_rate < 0):
            raise ValueError("arrival_rate must be a finite non-negative patients-per-minute rate")
        if isinstance(treatment_duration, tuple):
            if len(treatment_duration) != 2:
                raise ValueError("treatment_duration range must have two values")
            low, high = treatment_duration
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (low, high)) or low < 1 or high < low:
                raise ValueError("treatment_duration range must satisfy 1 <= min <= max")
        elif isinstance(treatment_duration, bool) or not isinstance(treatment_duration, int) or treatment_duration < 1:
            raise ValueError("treatment_duration must be a positive integer or (min, max)")

        self.scenario_name = scenario  # Mesa reserves ``scenario`` for Scenario objects.
        self.duration = duration
        self.current_minute = 0
        self.default_treatment_duration = treatment_duration
        self.classifier = classifier
        self.batch_size = batch_size
        self.urgency_weights = dict(DEFAULT_URGENCY_WEIGHTS if urgency_weights is None else urgency_weights)
        self.scheduler = AStarScheduler(self.urgency_weights, max_astar_expansions)
        self.patients: list[PatientAgent] = []
        self.waiting_queue: list[PatientAgent] = []
        self.maximum_queue_length = 0
        self.metrics = SimulationMetrics()
        self.last_schedule_result: dict[str, Any] = {
            "schedule": [], "weighted_wait_cost": 0.0, "expanded_states": 0,
            "execution_time_seconds": 0.0, "unschedulable": [], "optimal": True,
            "complete": True, "batch_patient_ids": [],
        }
        self._scheduled: dict[int, list[PatientRecord]] = {}
        self._next_generated_id = 1
        if patients is not None and arrival_schedule is not None:
            raise ValueError("Provide patients or arrival_schedule, not both")

        self.doctors = self._make_roster(doctor_roster, n_doctors, scenario)
        if arrival_rate is None and patients is None and arrival_schedule is None:
            arrival_rate = 0.35 if scenario == "surge" else 0.12
        self._generated_arrivals = arrival_rate is not None
        self.arrival_rate = float(arrival_rate or 0.0)

        if arrival_schedule is not None:
            for minute, records in arrival_schedule.items():
                self._validate_minute(minute)
                self._scheduled[minute] = list(records)
        elif patients is not None:
            for record in patients:
                if "arrival_time" not in record:
                    raise ValueError("Each patient record must include arrival_time")
                minute = record["arrival_time"]
                self._validate_minute(minute)
                self._scheduled.setdefault(minute, []).append(record)
        self.metrics.collect(self)

    def _make_roster(self, roster, n_doctors: int | None, scenario: str) -> list[DoctorAgent]:
        if roster is None:
            base = LIMITED_DOCTOR_ROSTER if scenario == "limited_staffing" else DEFAULT_DOCTOR_ROSTER
            if n_doctors is not None:
                if isinstance(n_doctors, bool) or not isinstance(n_doctors, int) or n_doctors < 1:
                    raise ValueError("n_doctors must be a positive integer")
                expanded = list(base)
                while len(expanded) < n_doctors:
                    expanded.append((f"D{len(expanded) + 1:03d}", "General Physician"))
                base = tuple(expanded[:n_doctors])
            roster = base
        elif n_doctors is not None:
            raise ValueError("Use doctor_roster or n_doctors, not both")

        result = []
        seen = set()
        for entry in roster:
            if isinstance(entry, Mapping):
                doctor_id, specialty = entry.get("doctor_id"), entry.get("specialty")
            else:
                try:
                    doctor_id, specialty = entry
                except (TypeError, ValueError) as exc:
                    raise ValueError("Roster entries must be (doctor_id, specialty) pairs or mappings") from exc
            if doctor_id in seen:
                raise ValueError(f"Duplicate doctor_id: {doctor_id}")
            seen.add(doctor_id)
            result.append(DoctorAgent(self, doctor_id, specialty))
        if not result:
            raise ValueError("doctor_roster must contain at least one doctor")
        return result

    @staticmethod
    def _validate_minute(minute: int) -> None:
        if isinstance(minute, bool) or not isinstance(minute, int) or minute < 0:
            raise ValueError("arrival times must be non-negative integer minutes")

    def _duration_for(self, record: PatientRecord) -> int:
        value = record.get("treatment_duration", self.default_treatment_duration)
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("treatment_duration range must have two values")
            low, high = value
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (low, high)) or low < 1 or high < low:
                raise ValueError("treatment duration range must satisfy 1 <= min <= max")
            return self.random.randint(low, high)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("treatment_duration must be a positive integer")
        return value

    def _arrive(self, record: PatientRecord) -> PatientAgent:
        data = dict(record)
        data.setdefault("arrival_time", self.current_minute)
        classification: dict[str, Any] = {}
        level = data.get("esi_level")
        if level is None and self.classifier is not None:
            classification = dict(self.classifier(data))
            level = classification.get("level")
        if level is None:
            raise ValueError("Patient needs esi_level or a classifier returning 'level'")
        level = validate_esi_level(level)
        patient_id = data.get("patient_id")
        if not patient_id:
            patient_id = f"P{self._next_generated_id:03d}"
            self._next_generated_id += 1
        patient = PatientAgent(
            self, patient_id=patient_id, age=data.get("age", 0),
            symptoms=data.get("symptoms", ""), arrival_time=data["arrival_time"],
            esi_level=level, required_specialty=data.get("required_specialty", "General Physician"),
            treatment_duration=self._duration_for(data), classification=classification,
        )
        if any(existing.patient_id == patient.patient_id for existing in self.patients):
            raise ValueError(f"Duplicate patient_id: {patient.patient_id}")
        patient.status = "waiting"
        self.patients.append(patient)
        if level == 1:
            # Bypass routine queues and doctors; this reserved emergency pathway is independent.
            patient.start_treatment(self.current_minute)
        else:
            self.waiting_queue.append(patient)
        return patient

    def _generate_arrivals(self) -> None:
        if not self._generated_arrivals or self.arrival_rate <= 0:
            return
        # A simple synthetic rate model: integer arrivals plus one Bernoulli remainder.
        count = int(self.arrival_rate)
        if self.random.random() < self.arrival_rate - count:
            count += 1
        count = min(count, 100)
        for _ in range(count):
            if self.scenario_name == "surge":
                level = self.random.choices([1, 2, 3, 4, 5], [0.12, 0.30, 0.30, 0.18, 0.10])[0]
            else:
                level = self.random.randint(2, 5)
            specialty = self.random.choice(SPECIALTIES)
            patient_id = f"P{self._next_generated_id:03d}"
            self._next_generated_id += 1
            self._arrive({"patient_id": patient_id, "age": self.random.randint(18, 90),
                          "symptoms": "Synthetic scenario", "arrival_time": self.current_minute,
                          "esi_level": level, "required_specialty": specialty})

    def _sort_queue(self) -> None:
        self.waiting_queue.sort(key=lambda p: (p.esi_level, p.arrival_time, p.patient_id))

    def _eligible_batch(self) -> list[PatientAgent]:
        covered = {normalize_specialty(doctor.specialty) for doctor in self.doctors}
        eligible = [patient for patient in self.waiting_queue
                    if normalize_specialty(patient.required_specialty) in covered]
        eligible.sort(key=lambda p: (p.esi_level, p.arrival_time, p.patient_id))
        return eligible[:self.batch_size]

    def _plan_waiting(self) -> dict[str, Any]:
        batch = self._eligible_batch()
        self.last_schedule_result = self.scheduler.schedule(batch, self.doctors, self.current_minute)
        covered = {normalize_specialty(d.specialty) for d in self.doctors}
        missing = [{"patient_id": p.patient_id,
                    "reason": f"No doctor covers required specialty {p.required_specialty!r}"}
                   for p in self.waiting_queue
                   if normalize_specialty(p.required_specialty) not in covered]
        self.last_schedule_result["unschedulable"].extend(missing)
        return self.last_schedule_result

    def _dispatch(self) -> None:
        self._plan_waiting()
        for row in self.last_schedule_result["schedule"]:
            if row["start_time"] > self.current_minute:
                continue
            doctor = next(d for d in self.doctors if d.doctor_id == row["doctor_id"])
            patient = next(p for p in self.waiting_queue if p.patient_id == row["patient_id"])
            if not doctor.is_available or not doctor.can_treat(patient):
                continue
            doctor.assign(patient, self.current_minute)
            self.waiting_queue.remove(patient)

    def _complete_treatments(self, end_time: int) -> bool:
        changed = False
        for doctor in self.doctors:
            patient = doctor.current_patient
            if patient is not None:
                doctor.busy_minutes += 1
                if doctor.available_at <= end_time:
                    patient.complete_treatment(end_time)
                    doctor.release(end_time)
                    changed = True
        for patient in self.patients:
            if patient.status == "emergency_treatment" and patient.treatment_start_time + patient.treatment_duration <= end_time:
                patient.complete_treatment(end_time)
                changed = True
        return changed

    def step(self) -> None:
        """Advance one simulated minute; arrivals and completion trigger replanning."""
        if self.current_minute >= self.duration:
            return
        for record in self._scheduled.pop(self.current_minute, []):
            self._arrive(record)
        self._generate_arrivals()
        self.maximum_queue_length = max(self.maximum_queue_length, len(self.waiting_queue))

        # Replan after arrivals and at every minute boundary. Active treatments
        # remain fixed; A* only plans waiting patients into present/future slots.
        self._dispatch()
        for patient in self.waiting_queue:
            patient.waiting_time = max(0, self.current_minute - patient.arrival_time)
        end_time = self.current_minute + 1
        completed = self._complete_treatments(end_time)
        self.current_minute = end_time
        for patient in self.waiting_queue:
            patient.waiting_time = max(0, self.current_minute - patient.arrival_time)
        self.maximum_queue_length = max(self.maximum_queue_length, len(self.waiting_queue))
        if completed:
            # Record that freed capacity caused a new plan. Actual assignment is
            # made at the next boundary so arrivals at that minute join the batch.
            self._plan_waiting()
        self.metrics.collect(self)

    def run_for(self, steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(min(steps, max(0, self.duration - self.current_minute))):
            self.step()

    def run_model(self) -> None:
        self.run_for(self.duration - self.current_minute)

    @property
    def doctor_utilisation(self) -> float:
        capacity = len(self.doctors) * self.current_minute
        return sum(doctor.busy_minutes for doctor in self.doctors) / capacity if capacity else 0.0

    @property
    def urgency_weighted_wait_cost(self) -> float:
        return sum(self.urgency_weights[p.esi_level] * p.waiting_time
                   for p in self.patients if p.esi_level in self.urgency_weights)

    @property
    def waiting_without_roster_coverage(self) -> int:
        covered = {normalize_specialty(d.specialty) for d in self.doctors}
        return sum(normalize_specialty(p.required_specialty) not in covered for p in self.waiting_queue)

    @property
    def waiting_without_available_specialist(self) -> int:
        return sum(not any(d.is_available and d.can_treat(p) for d in self.doctors)
                   for p in self.waiting_queue)

    def get_patient_queue(self) -> list[PatientAgent]:
        self._sort_queue()
        return list(self.waiting_queue)

    def get_doctor_roster(self) -> list[dict[str, Any]]:
        return [{"doctor_id": d.doctor_id, "specialty": d.specialty,
                 "availability_status": d.availability_status, "available": d.is_available,
                 "available_at": d.available_at,
                 "current_patient": d.current_patient.patient_id if d.current_patient else None,
                 "patients_treated": d.patients_treated} for d in self.doctors]

    def get_doctor_availability(self) -> list[dict[str, Any]]:
        return self.get_doctor_roster()

    def get_patient_statuses(self) -> list[dict[str, Any]]:
        return [{"patient_id": p.patient_id, "status": p.display_status,
                 "esi_level": p.esi_level, "required_specialty": p.required_specialty,
                 "waiting_time": p.waiting_time, "arrival_time": p.arrival_time,
                 "treatment_start_time": p.treatment_start_time,
                 "treatment_end_time": p.treatment_end_time} for p in self.patients]

    def get_metrics(self):
        return self.metrics.to_dataframe()

    def get_waiting_time_by_esi(self):
        return self.metrics.waiting_time_by_esi(self.patients)

    def get_patient_results(self):
        return self.metrics.patient_results(self.patients)

    def get_astar_results(self) -> dict[str, Any]:
        return dict(self.last_schedule_result)


# Keep the earlier class name so existing Member 3 imports continue to work.
TriageModel = EmergencyDepartmentModel
