# TriageBot – Member 2 Simulation

Educational, synthetic emergency-department simulation using Mesa 3.5.1. The
simulation accepts ESI levels as supplied inputs; it does **not** implement
clinical classification rules and is not clinically validated.

## Project files

```text
agents.py                    PatientAgent and specialty-bound DoctorAgent
model.py                     EmergencyDepartmentModel (TriageModel alias)
astar_scheduler.py           Standalone A* scheduling and exhaustive test oracle
baseline_schedulers.py       FCFS and ESI-priority comparison schedulers
metrics.py                   Pandas metrics and patient result exports
data/sample_patients.csv     Eight temporary synthetic patients
run_simulation.py            Sample simulation and scheduling comparison
tests/                       Agent, model, A*, and metrics tests
docs/astar_design.md         Search state, heuristic, and assumptions
```

## Setup and run

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_simulation.py
python -m unittest discover -s tests -v
```

One tick represents one simulated minute. The default model uses five doctors:
two General Physicians, one Surgeon, one Cardiologist, and one Neurologist.
Choose `scenario="limited_staffing"` or pass a custom `doctor_roster` to compare
staffing configurations. Normal and surge scenarios use synthetic arrivals when
no patient list or explicit schedule is supplied.

## Model API for the dashboard

```python
from model import EmergencyDepartmentModel

model = EmergencyDepartmentModel(
    patients=patient_records,  # optional; provide ESI levels or classifier
    duration=120,
    batch_size=8,
    scenario="normal",
)
model.step()                   # one simulated minute
model.run_for(30)              # advance 30 minutes
model.run_model()              # run to configured duration
model.get_patient_queue()
model.get_patient_statuses()
model.get_doctor_roster()
model.get_astar_results()
model.get_metrics()            # time-series DataFrame
model.get_waiting_time_by_esi()
model.get_patient_results()     # patient-level DataFrame
```

`TriageModel` remains an alias for `EmergencyDepartmentModel` for compatibility
with earlier Member 3 imports. Patient rows use `patient_id`, `age`,
`arrival_time`, `esi_level`, `required_specialty`, and `treatment_duration`;
`symptoms` is optional for simulation purposes.

Member 1 can connect `classify_patient(patient)` with
`EmergencyDepartmentModel(classifier=classify_patient)`. The callback returns a
mapping with `level`; `reason` and `trace` are retained when present. Specialty
routing remains the separate `required_specialty` field.

## Scheduling algorithms

`astar_scheduler.astar_schedule` minimizes the experimental urgency-weighted
waiting cost for a finite batch. FCFS and ESI-priority strategies are available
in `baseline_schedulers.py`; `compare_schedulers` runs all three against the
same patient and doctor inputs. Default weights are ESI 2: 4, ESI 3: 3, ESI 4:
2, ESI 5: 1. These are simulation weights, not ESI recommendations. ESI 1
patients use a separate immediate-emergency pathway and are excluded from all
ordinary scheduling queues.

Only doctors with the matching specialty may receive a patient. Patients with
no coverage stay queued and appear in the A* result's `unschedulable` list and
the model's staffing metrics. A* batches default to eight patients. If the
search reaches its expansion cap, the result says `optimal=False` and
`complete=False`; it does not claim to have found an optimum.

## Synthetic data and limitations

`data/sample_patients.csv` is temporary synthetic test data, not a real patient
dataset. Arrival patterns, treatment durations, doctor capacity, and urgency
weights are experiment assumptions. ESI 1 handling is represented as a separate
synthetic pathway without a routine doctor assignment. Do not use this model for
real patient-care decisions.
