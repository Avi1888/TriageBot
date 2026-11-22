"""Patient and doctor agents for the TriageBot simulation.

ESI classification is external input; this module contains no clinical rules.
"""

from __future__ import annotations

from typing import Any

from mesa import Agent

SPECIALTIES = ("General Physician", "Surgeon", "Cardiologist", "Neurologist")


def normalize_specialty(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("specialty must be a non-empty string")
    return " ".join(value.split()).casefold()


def validate_esi_level(level: int) -> int:
    """Validate a manually supplied or externally classified ESI level."""
    if isinstance(level, bool) or not isinstance(level, int) or level not in range(1, 6):
        raise ValueError("esi_level must be an integer from 1 to 5")
    return level


class PatientAgent(Agent):
    """A synthetic patient whose urgency and specialty are supplied as data."""

    def __init__(
        self,
        model: Any,
        patient_id: str,
        age: int,
        symptoms: str = "",
        arrival_time: int = 0,
        esi_level: int | None = None,
        treatment_duration: int = 1,
        required_specialty: str = "General Physician",
        classification: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model)
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("patient_id must be a non-empty string")
        if isinstance(age, bool) or not isinstance(age, int) or age < 0:
            raise ValueError("age must be a non-negative integer")
        if not isinstance(symptoms, str):
            raise ValueError("symptoms must be a string")
        if isinstance(arrival_time, bool) or not isinstance(arrival_time, int) or arrival_time < 0:
            raise ValueError("arrival_time must be a non-negative integer")
        if isinstance(treatment_duration, bool) or not isinstance(treatment_duration, int) or treatment_duration < 1:
            raise ValueError("treatment_duration must be a positive integer")
        self.required_specialty = " ".join(required_specialty.split()) if isinstance(required_specialty, str) else required_specialty
        normalize_specialty(self.required_specialty)
        if esi_level is not None:
            esi_level = validate_esi_level(esi_level)

        self.patient_id = patient_id.strip()
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

    @property
    def treatment_end_time(self) -> int | None:
        """Compatibility alias used by the project data contract."""
        return self.treatment_completion_time

    @treatment_end_time.setter
    def treatment_end_time(self, value: int | None) -> None:
        self.treatment_completion_time = value

    @property
    def display_status(self) -> str:
        return {"waiting": "Waiting", "being_treated": "In Treatment",
                "emergency_treatment": "In Treatment", "completed": "Completed"}.get(self.status, self.status)

    def start_treatment(self, start_time: int, doctor_id: str | None = None) -> None:
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
        """TriageModel advances patient state centrally in a fixed order."""


class DoctorAgent(Agent):
    """A doctor with one fixed specialty and a single treatment slot."""

    def __init__(self, model: Any, doctor_id: str, specialty: str = "General Physician") -> None:
        super().__init__(model)
        if not isinstance(doctor_id, str) or not doctor_id.strip():
            raise ValueError("doctor_id must be a non-empty string")
        self.doctor_id = doctor_id.strip()
        self.specialty = " ".join(specialty.split()) if isinstance(specialty, str) else specialty
        normalize_specialty(self.specialty)
        self.current_patient: PatientAgent | None = None
        self.available_at = 0
        self.patients_treated = 0
        self.busy_minutes = 0

    @property
    def is_available(self) -> bool:
        return self.current_patient is None

    @property
    def availability_status(self) -> str:
        return "Available" if self.is_available else "Treating"

    @property
    def availability(self) -> bool:
        return self.is_available

    @property
    def treatment_status(self) -> str:
        return "available" if self.is_available else "treating"

    def can_treat(self, patient: PatientAgent) -> bool:
        return normalize_specialty(self.specialty) == normalize_specialty(patient.required_specialty)

    def assign(self, patient: PatientAgent, start_time: int | None = None) -> None:
        if not self.is_available:
            raise ValueError(f"{self.doctor_id} is already treating a patient")
        if not self.can_treat(patient):
            raise ValueError(f"{self.doctor_id} cannot treat specialty {patient.required_specialty!r}")
        start_time = self.available_at if start_time is None else start_time
        patient.start_treatment(start_time, self.doctor_id)
        self.current_patient = patient
        self.available_at = start_time + patient.treatment_duration

    def release(self, current_time: int | None = None) -> PatientAgent | None:
        patient = self.current_patient
        if patient is not None:
            self.patients_treated += 1
        self.current_patient = None
        if current_time is not None:
            self.available_at = max(self.available_at, current_time)
        return patient

    def step(self) -> None:
        """TriageModel advances doctor state centrally in a fixed order."""
