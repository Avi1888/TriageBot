"""Run the bundled synthetic sample through the model and schedulers."""

from __future__ import annotations

import csv
from pathlib import Path

from baseline_schedulers import compare_schedulers
from model import DEFAULT_DOCTOR_ROSTER, EmergencyDepartmentModel


def load_sample() -> list[dict]:
    sample = Path(__file__).parent / "data" / "sample_patients.csv"
    with sample.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for column in ("age", "arrival_time", "esi_level", "treatment_duration"):
            row[column] = int(row[column])
    return rows


def main() -> None:
    patients = load_sample()
    comparison = compare_schedulers(patients, DEFAULT_DOCTOR_ROSTER)
    print("Pre-run scheduling comparison (weighted waiting cost):")
    for name, result in comparison.items():
        quality = "optimal" if result["optimal"] else "baseline"
        print(f"  {name:12s} {result['weighted_wait_cost']:>6.1f} ({quality})")

    model = EmergencyDepartmentModel(patients=patients, duration=120, scenario="normal")
    model.run_model()
    completed = int(model.get_metrics().iloc[-1]["number_completed"])
    print(f"\nSimulation finished at minute {model.current_minute}.")
    print(f"Patients: {len(model.patients)}; completed: {completed}; "
          f"still waiting: {len(model.waiting_queue)}.")
    print(f"Doctor utilisation: {model.doctor_utilisation:.1%}")
    print("\nPatient outcomes:")
    print(model.get_patient_results()[
        ["patient_id", "esi_level", "required_specialty", "waiting_time", "status", "doctor_id"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
