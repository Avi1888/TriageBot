"""Mesa agents used by the educational emergency-department simulation.

ESI levels are inputs to this module. This file deliberately contains no
clinical assessment or ESI classification rules.
"""

from __future__ import annotations

from typing import Any

from mesa import Agent


class PatientAgent(Agent):
    """A synthetic patient moving through the simulated department."""

    def __init__(
        self,
        model: Any,
        patient_id: str,
        age: int,
        symptoms: str,
        arrival_time: int,
        esi_level: int | None = None,
        treatment_duration: int = 1,
        classification: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model)
        if not patient_id or not isinstance(patient_id, str):
            raise ValueError("patient_id must be a non-empty string")
        if isinstance(age, bool) or not isinstance(age, int) or age < 0:
            raise ValueError("age must be a non-negative integer")
        if not isinstance(symptoms, str):
            raise ValueError("symptoms must be a string")
        if isinstance(arrival_time, bool) or not isinstance(arrival_time, int) or arrival_time < 0:
            raise ValueError("arrival_time must be a non-negative integer")
        if isinstance(treatment_duration, bool) or not isinstance(treatment_duration, int) or treatment_duration < 1:
            raise ValueError("treatment_duration must be a positive integer")
        if esi_level is not None:
            esi_level = validate_esi_level(esi_level)

        self.patient_id = patient_id
        self.age = age
        self.symptoms = symptoms
        self.arrival_time = arrival_time
        self.esi_level = esi_level
        self.classification_reason = (classification or {}).get("reason")
        self.classification_trace = list((classification or {}).get("trace", []))
        self.waiting_time = 0
        self.treatment_duration = treatment_duration
        self.treatment_start_time: int | None = None
        self.treatment_completion_time: int | None = None
        self.status = "scheduled"
        self.doctor_id: str | None = None

    def start_treatment(self, start_time: int, doctor_id: str | None = None) -> None:
        """Start ordinary or emergency treatment at the supplied minute."""
        if self.status != "waiting":
            raise ValueError("Only waiting patients can start treatment")
        self.waiting_time = max(0, start_time - self.arrival_time)
        self.treatment_start_time = start_time
        self.doctor_id = doctor_id
        self.status = "emergency_treatment" if self.esi_level == 1 else "being_treated"

    def complete_treatment(self, completion_time: int) -> None:
        self.treatment_completion_time = completion_time
        self.status = "completed"

    def step(self) -> None:
        """Agents are advanced centrally by TriageModel for deterministic order."""


class DoctorAgent(Agent):
    """A doctor resource that can treat at most one patient at a time."""

    def __init__(self, model: Any, doctor_id: str) -> None:
        super().__init__(model)
        self.doctor_id = doctor_id
        self.current_patient: PatientAgent | None = None
        self.busy_minutes = 0

    @property
    def is_available(self) -> bool:
        return self.current_patient is None

    @property
    def availability(self) -> bool:
        """Alias suitable for dashboard display."""
        return self.is_available

    @property
    def treatment_status(self) -> str:
        return "available" if self.is_available else "treating"

    def assign(self, patient: PatientAgent) -> None:
        if not self.is_available:
            raise ValueError(f"{self.doctor_id} is already treating a patient")
        self.current_patient = patient

    def release(self) -> None:
        self.current_patient = None

    def step(self) -> None:
        """Doctor state is advanced centrally by TriageModel."""


def validate_esi_level(level: int) -> int:
    """Validate a manually supplied or externally classified ESI level."""
    if isinstance(level, bool) or not isinstance(level, int) or level not in range(1, 6):
        raise ValueError("esi_level must be an integer from 1 to 5")
    return level
