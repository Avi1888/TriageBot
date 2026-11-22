"""Unit tests for the independent Mesa simulation module.

Run with: python -m unittest discover -s tests -v
"""

import unittest

from agents import PatientAgent
from model import TriageModel


def patient(patient_id, arrival=0, esi=3, duration=2):
    return {"patient_id": patient_id, "age": 45, "symptoms": "Synthetic example",
            "arrival_time": arrival, "esi_level": esi, "treatment_duration": duration}


class SimulationTests(unittest.TestCase):
    def test_patient_creation_and_esi_validation(self):
        model = TriageModel(duration=5)
        p = PatientAgent(model, "P001", 45, "Example", 0, 3, 2)
        self.assertEqual(p.status, "scheduled")
        with self.assertRaises(ValueError):
            PatientAgent(model, "P002", 45, "Example", 0, 6, 2)

    def test_doctor_availability_and_assignment(self):
        model = TriageModel(n_doctors=1, duration=3)
        doctor = model.doctors[0]
        self.assertTrue(doctor.is_available)
        p = PatientAgent(model, "P001", 45, "Example", 0, 3, 2)
        p.status = "waiting"
        doctor.assign(p)
        self.assertFalse(doctor.availability)
        self.assertEqual(doctor.treatment_status, "treating")

    def test_arrivals_and_queue_priority(self):
        model = TriageModel(patients=[patient("low", esi=5), patient("urgent", esi=2)],
                            n_doctors=1, duration=5)
        model.step()
        self.assertEqual(model.doctors[0].current_patient.patient_id, "urgent")
        self.assertEqual([p.patient_id for p in model.get_patient_queue()], ["low"])

    def test_equal_priority_uses_earlier_arrival(self):
        model = TriageModel(patients=[patient("later", arrival=1), patient("first", arrival=0)],
                            n_doctors=1, duration=4)
        model.step()
        self.assertEqual(model.doctors[0].current_patient.patient_id, "first")

    def test_treatment_completes_and_doctor_releases(self):
        model = TriageModel(patients=[patient("P001", duration=2)], n_doctors=1, duration=5)
        model.run_for(2)
        self.assertEqual(model.patients[0].status, "completed")
        self.assertEqual(model.patients[0].treatment_completion_time, 2)
        self.assertTrue(model.doctors[0].is_available)

    def test_waiting_time_for_delayed_patient(self):
        records = [patient("first", esi=5, duration=3), patient("urgent", arrival=1, esi=2)]
        model = TriageModel(patients=records, n_doctors=1, duration=8)
        model.step()
        model.step()
        model.step()
        model.step()
        self.assertEqual(model.doctors[0].current_patient.patient_id, "urgent")
        self.assertEqual(model.patients[1].waiting_time, 2)

    def test_esi_one_bypasses_ordinary_queue(self):
        model = TriageModel(patients=[patient("critical", esi=1, duration=2),
                                      patient("routine", esi=5)], n_doctors=1, duration=4)
        model.step()
        critical = next(p for p in model.patients if p.patient_id == "critical")
        self.assertEqual(critical.status, "emergency_treatment")
        self.assertEqual([p.patient_id for p in model.get_patient_queue()], [])
        self.assertEqual(model.doctors[0].current_patient.patient_id, "routine")

    def test_future_classifier_callback(self):
        calls = []
        def classify(record):
            calls.append(record["patient_id"])
            return {"level": 4, "reason": "Example", "trace": ["Example trace"]}
        model = TriageModel(patients=[{"patient_id": "P001", "age": 20, "symptoms": "x", "arrival_time": 0}],
                            classifier=classify, duration=2)
        model.step()
        self.assertEqual(calls, ["P001"])
        self.assertEqual(model.patients[0].esi_level, 4)
        self.assertEqual(model.patients[0].classification_reason, "Example")

    def test_metrics_and_dataframe_exports(self):
        model = TriageModel(patients=[patient("P001", duration=1)], duration=3)
        model.run_model()
        metrics = model.get_metrics()
        results = model.get_patient_results()
        self.assertEqual(len(metrics), 4)
        self.assertEqual(int(metrics.iloc[-1]["completed_patients"]), 1)
        self.assertEqual(int(results.iloc[0]["waiting_time"]), 0)
        self.assertEqual(list(model.metrics.waiting_time_by_esi(model.patients)["esi_level"]), [1, 2, 3, 4, 5])

    def test_runs_all_scenarios_and_multiple_doctors(self):
        normal = TriageModel(scenario="normal", n_doctors=2, arrival_rate=0.5, duration=10, seed=1)
        surge = TriageModel(scenario="surge", duration=10, seed=1)
        limited = TriageModel(scenario="limited_staffing", duration=10, seed=1)
        normal.run_model()
        surge.run_model()
        limited.run_model()
        self.assertEqual(len(normal.doctors), 2)
        self.assertEqual(len(limited.doctors), 1)
        self.assertEqual(normal.time, 10)
        self.assertGreater(len(surge.patients), 0)

    def test_validation_and_step_limit(self):
        with self.assertRaises(ValueError):
            TriageModel(n_doctors=0)
        model = TriageModel(duration=2)
        model.run_for(10)
        self.assertEqual(model.time, 2)


if __name__ == "__main__":
    unittest.main()
