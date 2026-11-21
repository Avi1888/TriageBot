"""Performance metrics for the emergency-department model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from model import EmergencyDepartmentModel


class SimulationMetrics:
    """Collect minute-level metrics and patient-level result frames."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def collect(self, model: "EmergencyDepartmentModel") -> None:
        patients = model.patients
        waiting = [p for p in patients if p.status == "waiting"]
        treating = [p for p in patients if p.status in ("being_treated", "emergency_treatment")]
        started = [p for p in patients if p.treatment_start_time is not None]
        completed = [p for p in patients if p.status == "completed"]
        waits = [p.waiting_time for p in started]
        self._records.append({
            "time": model.current_minute,
            "queue_length": len(model.waiting_queue),
            "number_waiting": len(waiting),
            "number_being_treated": len(treating),
            "number_completed": len(completed),
            "completed_patients": len(completed),  # compatibility alias
            "total_patients_treated": len(started),
            "patient_throughput": len(completed),
            "total_patient_throughput": len(completed),  # compatibility alias
            "average_waiting_time": sum(waits) / len(waits) if waits else 0.0,
            "maximum_patient_waiting_time": max((p.waiting_time for p in patients), default=0),
            "urgency_weighted_wait_cost": model.urgency_weighted_wait_cost,
            "doctor_utilisation": model.doctor_utilisation,
            "maximum_queue_length": model.maximum_queue_length,
            "astar_execution_time_seconds": model.last_schedule_result.get("execution_time_seconds", 0.0),
            "astar_expanded_states": model.last_schedule_result.get("expanded_states", 0),
            "waiting_without_available_specialist": model.waiting_without_available_specialist,
            "waiting_without_roster_coverage": model.waiting_without_roster_coverage,
        })

    @staticmethod
    def waiting_time_by_esi(patients: list) -> pd.DataFrame:
        rows = []
        for level in range(1, 6):
            completed_or_started = [p for p in patients if p.esi_level == level and p.treatment_start_time is not None]
            waits = sorted(p.waiting_time for p in completed_or_started)
            mid = len(waits) // 2
            median = (float(waits[mid]) if len(waits) % 2 else
                      (waits[mid - 1] + waits[mid]) / 2 if waits else 0.0)
            all_in_level = [p for p in patients if p.esi_level == level]
            observed_waits = [p.waiting_time for p in all_in_level]
            rows.append({"esi_level": level,
                         "average_waiting_time": sum(observed_waits) / len(observed_waits) if observed_waits else 0.0,
                         "average_wait_for_started_patients": sum(waits) / len(waits) if waits else 0.0,
                         "median_waiting_time": median,
                         "patients_waiting": sum(p.status == "waiting" for p in all_in_level),
                         "patients_started": len(waits)})
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records).copy()

    @staticmethod
    def patient_results(patients: list) -> pd.DataFrame:
        columns = ["patient_id", "age", "symptoms", "arrival_time", "esi_level",
                   "required_specialty", "status", "waiting_time", "treatment_duration",
                   "treatment_start_time", "treatment_end_time", "doctor_id",
                   "classification_reason", "classification_trace"]
        rows = []
        for patient in patients:
            rows.append({
                "patient_id": patient.patient_id, "age": patient.age, "symptoms": patient.symptoms,
                "arrival_time": patient.arrival_time, "esi_level": patient.esi_level,
                "required_specialty": patient.required_specialty, "status": patient.display_status,
                "waiting_time": patient.waiting_time, "treatment_duration": patient.treatment_duration,
                "treatment_start_time": patient.treatment_start_time,
                "treatment_end_time": patient.treatment_end_time, "doctor_id": patient.doctor_id,
                "classification_reason": patient.classification_reason,
                "classification_trace": patient.classification_trace,
            })
        return pd.DataFrame(rows, columns=columns)


def collect_metrics(model: "EmergencyDepartmentModel") -> pd.DataFrame:
    """Return the model's collected metric time series."""
    return model.metrics.to_dataframe()
