"""Data-frame snapshots and calculations for TriageModel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from model import TriageModel


class SimulationMetrics:
    """Collect model-level time series and summarize patient outcomes."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def collect(self, model: "TriageModel") -> None:
        patients = model.patients
        waiting = [p for p in patients if p.status == "waiting"]
        treating = [p for p in patients if p.status in ("being_treated", "emergency_treatment")]
        completed = [p for p in patients if p.status == "completed"]
        self._records.append({
            "time": model.current_minute,
            "queue_length": len(model.waiting_queue),
            "waiting_patients": len(waiting),
            "patients_being_treated": len(treating),
            "completed_patients": len(completed),
            "total_patient_throughput": len(completed),
            "average_waiting_time": self.average_waiting_time(patients),
            "median_waiting_time": self.median_waiting_time(patients),
            "doctor_utilisation": model.doctor_utilisation,
            "maximum_queue_length": model.maximum_queue_length,
        })

    @staticmethod
    def average_waiting_time(patients: list) -> float:
        waits = [p.waiting_time for p in patients if p.treatment_start_time is not None]
        return sum(waits) / len(waits) if waits else 0.0

    @staticmethod
    def median_waiting_time(patients: list) -> float:
        waits = sorted(p.waiting_time for p in patients if p.treatment_start_time is not None)
        if not waits:
            return 0.0
        mid = len(waits) // 2
        return float(waits[mid]) if len(waits) % 2 else (waits[mid - 1] + waits[mid]) / 2

    @staticmethod
    def waiting_time_by_esi(patients: list) -> pd.DataFrame:
        rows = [
            {"esi_level": level,
             "average_waiting_time": SimulationMetrics.average_waiting_time(
                 [p for p in patients if p.esi_level == level]
             ),
             "median_waiting_time": SimulationMetrics.median_waiting_time(
                 [p for p in patients if p.esi_level == level]
             ),
             "patients_treated": sum(
                 1 for p in patients if p.esi_level == level and p.treatment_start_time is not None
             )}
            for level in range(1, 6)
        ]
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records).copy()

    @staticmethod
    def patient_results(patients: list) -> pd.DataFrame:
        columns = [
            "patient_id", "age", "symptoms", "arrival_time", "esi_level", "status",
            "waiting_time", "treatment_duration", "treatment_start_time",
            "treatment_completion_time", "doctor_id", "classification_reason",
            "classification_trace",
        ]
        return pd.DataFrame([{column: getattr(p, column) for column in columns} for p in patients],
                            columns=columns)
