"""Integration tests for arrivals, A* dispatch, specialty routing and treatment."""

import unittest

from agents import DoctorAgent, PatientAgent
from model import DEFAULT_DOCTOR_ROSTER, EmergencyDepartmentModel, TriageModel


def patient(patient_id, arrival=0, esi=3, specialty="General Physician", duration=2):
    return {"patient_id": patient_id, "age": 45, "symptoms": "Synthetic example",
            "arrival_time": arrival, "esi_level": esi,
            "required_specialty": specialty, "treatment_duration": duration}


class SimulationTests(unittest.TestCase):
    def test_five_doctor_default_roster_and_legacy_alias(self):
        model = EmergencyDepartmentModel(duration=3)
        self.assertEqual(len(model.doctors), 5)
        self.assertEqual(model.doctors[0].specialty, "General Physician")
        self.assertEqual(model.doctors[2].specialty, "Surgeon")
        self.assertEqual(model.doctors[3].specialty, "Cardiologist")
        self.assertEqual(model.doctors[4].specialty, "Neurologist")
        self.assertIs(TriageModel, EmergencyDepartmentModel)
        self.assertEqual(DEFAULT_DOCTOR_ROSTER[0][1], "General Physician")

    def test_patient_and_doctor_creation(self):
        model = EmergencyDepartmentModel(n_doctors=1, duration=3)
        p = PatientAgent(model, "P001", 45, "Example", 0, 3, 2,
                         required_specialty="General Physician")
        d = DoctorAgent(model, "D999", "General Physician")
        self.assertEqual(p.treatment_duration, 2)
        self.assertEqual(p.required_specialty, "General Physician")
        self.assertEqual(d.availability_status, "Available")

    def test_patient_esi_input_validated(self):
        model = EmergencyDepartmentModel(duration=3)
        with self.assertRaises(ValueError):
            PatientAgent(model, "P001", 45, "x", 0, 6)

    def test_priority_and_specialty_constrained_allocation(self):
        rows = [patient("low", esi=5), patient("cardiac", esi=2, specialty="Cardiologist"),
                patient("urgent_gp", esi=2)]
        model = EmergencyDepartmentModel(patients=rows, n_doctors=2, duration=5)
        # n_doctors=2 creates two GPs, so cardiology waits for a compatible doctor.
        model.step()
        treating = {d.current_patient.patient_id: d.specialty for d in model.doctors if d.current_patient}
        self.assertEqual(treating, {"urgent_gp": "General Physician", "low": "General Physician"})
        self.assertEqual([p.patient_id for p in model.get_patient_queue()], ["cardiac"])

    def test_no_wrong_specialty_assignment(self):
        model = EmergencyDepartmentModel(patients=[patient("cardiac", esi=2, specialty="Cardiologist")],
                                         n_doctors=1, duration=4)
        model.step()
        self.assertIsNone(model.doctors[0].current_patient)
        self.assertEqual(model.get_patient_queue()[0].patient_id, "cardiac")
        with self.assertRaises(ValueError):
            model.doctors[0].assign(model.get_patient_queue()[0])

    def test_busy_doctor_patient_waits_then_starts(self):
        rows = [patient("first", duration=2), patient("second", arrival=1, esi=2, duration=1)]
        model = EmergencyDepartmentModel(patients=rows, n_doctors=1, duration=5)
        model.step()  # first starts at t=0
        model.step()  # second arrives at t=1, cannot interrupt first
        self.assertEqual(model.patients[0].status, "completed")
        self.assertEqual(model.get_patient_queue()[0].patient_id, "second")
        model.step()  # second begins at t=2 after the first doctor slot is released
        second = next(p for p in model.patients if p.patient_id == "second")
        self.assertEqual(second.treatment_start_time, 2)
        self.assertEqual(second.waiting_time, 1)

    def test_missing_specialty_coverage_reported_and_patient_stays_queued(self):
        model = EmergencyDepartmentModel(patients=[patient("neuro", specialty="Neurologist")],
                                         n_doctors=1, duration=3)
        model.run_model()
        self.assertEqual(model.get_patient_queue()[0].patient_id, "neuro")
        result = model.get_astar_results()
        self.assertEqual(result["unschedulable"][0]["patient_id"], "neuro")
        self.assertEqual(model.get_metrics().iloc[-1]["waiting_without_roster_coverage"], 1)

    def test_level_one_uses_immediate_pathway(self):
        model = EmergencyDepartmentModel(patients=[patient("critical", esi=1, duration=2),
                                                   patient("routine", esi=5)], n_doctors=1, duration=4)
        model.step()
        critical = next(p for p in model.patients if p.patient_id == "critical")
        self.assertEqual(critical.status, "emergency_treatment")
        self.assertEqual([p.patient_id for p in model.get_patient_queue()], [])
        self.assertEqual(model.doctors[0].current_patient.patient_id, "routine")

    def test_classifier_callback_remains_separate(self):
        def classify_patient(record):
            return {"level": 4, "reason": "test-only result", "trace": ["external module"]}
        record = {"patient_id": "from-member-1", "age": 23, "symptoms": "synthetic",
                  "arrival_time": 0, "required_specialty": "Surgeon", "treatment_duration": 1}
        model = EmergencyDepartmentModel(patients=[record], classifier=classify_patient, duration=2)
        model.step()
        self.assertEqual(model.patients[0].esi_level, 4)
        self.assertEqual(model.patients[0].classification_reason, "test-only result")
        self.assertEqual(model.patients[0].classification_trace, ["external module"])

    def test_dynamic_replanning_does_not_change_active_patient(self):
        model = EmergencyDepartmentModel(patients=[patient("active", duration=3),
                                                   patient("waiting", arrival=1, esi=2)],
                                         n_doctors=1, duration=6)
        model.step()
        model.step()
        self.assertEqual(model.doctors[0].current_patient.patient_id, "active")
        model.run_for(2)
        self.assertEqual(model.doctors[0].current_patient.patient_id, "waiting")

    def test_arrivals_at_multiple_times_and_execution_duration(self):
        model = EmergencyDepartmentModel(
            arrival_schedule={0: [patient("A")], 3: [patient("B", arrival=3)]},
            n_doctors=2, duration=5, treatment_duration=1,
        )
        model.run_model()
        self.assertEqual([p.arrival_time for p in model.patients], [0, 3])
        self.assertEqual(model.current_minute, 5)

    def test_synthetic_arrival_rates_and_staffing_scenarios(self):
        normal = EmergencyDepartmentModel(scenario="normal", arrival_rate=2.0,
                                          duration=4, seed=3)
        surge = EmergencyDepartmentModel(scenario="surge", arrival_rate=1.0,
                                         duration=4, seed=3)
        limited = EmergencyDepartmentModel(scenario="limited_staffing", duration=4)
        normal.run_model()
        surge.run_model()
        self.assertEqual(len(normal.patients), 8)
        self.assertEqual(len(normal.doctors), 5)
        self.assertEqual(len(surge.patients), 4)
        self.assertEqual(len(limited.doctors), 1)
        self.assertTrue(all(1 <= patient.esi_level <= 5 for patient in surge.patients))

    def test_custom_roster_and_unschedulable_wait_metric(self):
        model = EmergencyDepartmentModel(
            patients=[patient("neuro", specialty="Neurologist")],
            doctor_roster=[("D1", "General Physician"), ("D2", "Cardiologist")],
            duration=2,
        )
        model.step()
        latest = model.get_metrics().iloc[-1]
        self.assertEqual(latest["waiting_without_roster_coverage"], 1)
        self.assertEqual(latest["waiting_without_available_specialist"], 1)


if __name__ == "__main__":
    unittest.main()
