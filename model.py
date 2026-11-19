"""Minute-stepped emergency department simulation using Mesa 3."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from mesa import Model

from agents import DoctorAgent, PatientAgent, validate_esi_level
from metrics import SimulationMetrics

PatientRecord = Mapping[str, Any]
Classifier = Callable[[PatientRecord], Mapping[str, Any]]
SCENARIOS = ("normal", "surge", "limited_staffing")


class TriageModel(Model):
    """Simulate patient arrivals, queueing, doctor service, and outcomes.

    ``time`` counts simulated minutes. A patient can supply ``esi_level`` or
    the caller can pass a future ``classify_patient(patient)`` callback that
    returns ``{"level": int, "reason": str, "trace": list}``. No ESI logic
    is implemented here.

    ESI 1 uses a separate immediate emergency-handling pathway and therefore
    does not consume an ordinary queue position or routine doctor slot. This
    is a simplifying simulation assumption, not a clinical workflow claim.
    """

    def __init__(
        self,
        patients: Iterable[PatientRecord] | None = None,
        n_doctors: int | None = None,
        arrival_schedule: Mapping[int, Iterable[PatientRecord]] | None = None,
        arrival_rate: float | None = None,
        treatment_duration: int | tuple[int, int] = (10, 30),
        duration: int = 120,
        scenario: str = "normal",
        classifier: Classifier | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(rng=seed)
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if n_doctors is not None and (isinstance(n_doctors, bool) or not isinstance(n_doctors, int) or n_doctors < 1):
            raise ValueError("n_doctors must be a positive integer")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
            raise ValueError("duration must be a non-negative integer")
        if arrival_rate is not None and (not 0 <= arrival_rate <= 1):
            raise ValueError("arrival_rate must be between 0 and 1 patients per minute")
        if isinstance(treatment_duration, tuple):
            low, high = treatment_duration
            if low < 1 or high < low:
                raise ValueError("treatment_duration range must satisfy 1 <= low <= high")
        elif isinstance(treatment_duration, bool) or not isinstance(treatment_duration, int) or treatment_duration < 1:
            raise ValueError("treatment_duration must be a positive integer or (min, max)")

        # Mesa 3 reserves ``Model.scenario`` for Scenario instances.
        self.scenario_name = scenario
        if n_doctors is None:
            n_doctors = 1 if scenario == "limited_staffing" else 3
        self.duration = duration
        self.default_treatment_duration = treatment_duration
        self.classifier = classifier
        # Mesa 3 owns ``Model.time`` as a timedelta; keep integer minutes here.
        self.current_minute = 0
        self.patients: list[PatientAgent] = []
        self.waiting_queue: list[PatientAgent] = []
        self.maximum_queue_length = 0
        self.metrics = SimulationMetrics()
        self.doctors = [DoctorAgent(self, f"D{i + 1:03d}") for i in range(n_doctors)]
        self._scheduled: dict[int, list[PatientRecord]] = {}
        if arrival_rate is None and patients is None and arrival_schedule is None:
            arrival_rate = 0.35 if scenario == "surge" else 0.12
        self._generated_arrivals = arrival_rate is not None
        self.arrival_rate = arrival_rate
        self._next_generated_id = 1

        # Scenario presets can be overridden explicitly by passing values.
        if arrival_schedule is not None and patients is not None:
            raise ValueError("Provide patients or arrival_schedule, not both")
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

    @staticmethod
    def _validate_minute(minute: int) -> None:
        if isinstance(minute, bool) or not isinstance(minute, int) or minute < 0:
            raise ValueError("arrival times must be non-negative integer minutes")

    def _duration_for(self, record: PatientRecord) -> int:
        value = record.get("treatment_duration", self.default_treatment_duration)
        if isinstance(value, tuple):
            low, high = value
            if low < 1 or high < low:
                raise ValueError("treatment duration range must satisfy 1 <= low <= high")
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
        if level is not None:
            level = validate_esi_level(level)
        if level is None:
            raise ValueError("Patient needs esi_level or a classifier that returns 'level'")
        if not data.get("patient_id"):
            data["patient_id"] = f"P{self._next_generated_id:03d}"
            self._next_generated_id += 1
        patient = PatientAgent(
            self, patient_id=data["patient_id"], age=data.get("age", 0),
            symptoms=data.get("symptoms", ""), arrival_time=data["arrival_time"],
            esi_level=level, treatment_duration=self._duration_for(data),
            classification=classification,
        )
        patient.status = "waiting"
        self.patients.append(patient)
        if patient.esi_level == 1:
            # Immediate emergency pathway bypasses the routine waiting queue.
            patient.start_treatment(self.current_minute)
        else:
            self.waiting_queue.append(patient)
        return patient

    def _generate_arrivals(self) -> None:
        if not self._generated_arrivals or not self.arrival_rate:
            return
        count = 0
        while self.random.random() < self.arrival_rate and count < 100:
            # Synthetic classification levels are supplied as simulation inputs,
            # never derived from symptoms. Surge skews the synthetic mix upward.
            level = self.random.choices([1, 2, 3, 4, 5], [0.12, 0.30, 0.30, 0.18, 0.10])[0] if self.scenario_name == "surge" else self.random.randint(2, 5)
            self._arrive({"patient_id": f"P{self._next_generated_id:03d}", "age": self.random.randint(18, 90),
                          "symptoms": "Synthetic scenario", "arrival_time": self.current_minute, "esi_level": level})
            self._next_generated_id += 1
            count += 1

    def _sort_queue(self) -> None:
        self.waiting_queue.sort(key=lambda patient: (patient.esi_level, patient.arrival_time, patient.unique_id))

    def _allocate_doctors(self) -> None:
        self._sort_queue()
        for doctor in self.doctors:
            if doctor.is_available and self.waiting_queue:
                patient = self.waiting_queue.pop(0)
                doctor.assign(patient)
                patient.start_treatment(self.current_minute, doctor.doctor_id)

    def _complete_treatments(self, end_time: int) -> None:
        for doctor in self.doctors:
            patient = doctor.current_patient
            if patient is not None and patient.treatment_start_time + patient.treatment_duration <= end_time:
                patient.complete_treatment(end_time)
                doctor.release()
        for patient in self.patients:
            if patient.status == "emergency_treatment" and patient.treatment_start_time + patient.treatment_duration <= end_time:
                patient.complete_treatment(end_time)

    def step(self) -> None:
        """Advance one simulated minute and collect an end-of-minute snapshot."""
        if self.current_minute >= self.duration:
            return
        for record in self._scheduled.pop(self.current_minute, []):
            self._arrive(record)
        self._generate_arrivals()
        self._allocate_doctors()
        for patient in self.waiting_queue:
            patient.waiting_time = self.current_minute - patient.arrival_time
        for doctor in self.doctors:
            if not doctor.is_available:
                doctor.busy_minutes += 1
        end_time = self.current_minute + 1
        self._complete_treatments(end_time)
        self.current_minute = end_time
        self.maximum_queue_length = max(self.maximum_queue_length, len(self.waiting_queue))
        self.metrics.collect(self)

    def run_for(self, steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(min(steps, max(0, self.duration - self.current_minute))):
            self.step()

    def run_model(self) -> None:
        """Run until the configured simulation duration."""
        self.run_for(self.duration - self.current_minute)

    @property
    def doctor_utilisation(self) -> float:
        elapsed_capacity = len(self.doctors) * self.current_minute
        return sum(doctor.busy_minutes for doctor in self.doctors) / elapsed_capacity if elapsed_capacity else 0.0

    def get_patient_queue(self) -> list[PatientAgent]:
        self._sort_queue()
        return list(self.waiting_queue)

    def get_doctor_availability(self) -> list[dict[str, Any]]:
        return [{"doctor_id": d.doctor_id, "available": d.is_available,
                 "patient_id": d.current_patient.patient_id if d.current_patient else None,
                 "treatment_status": d.treatment_status} for d in self.doctors]

    def get_patient_statuses(self) -> list[dict[str, Any]]:
        return [{"patient_id": p.patient_id, "status": p.status, "esi_level": p.esi_level,
                 "waiting_time": p.waiting_time, "arrival_time": p.arrival_time,
                 "treatment_start_time": p.treatment_start_time,
                 "treatment_completion_time": p.treatment_completion_time} for p in self.patients]

    def get_metrics(self):
        return self.metrics.to_dataframe()

    def get_waiting_time_by_esi(self):
        return self.metrics.waiting_time_by_esi(self.patients)

    def get_patient_results(self):
        return self.metrics.patient_results(self.patients)
