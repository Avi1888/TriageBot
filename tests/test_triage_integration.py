import unittest

from triage_integration import create_triage_model


class TriageIntegrationTests(unittest.TestCase):
    def test_member_one_record_is_classified_and_simulated(self):
        record = {
            "patient_id": "M1-001", "age": 45, "chief_complaint": "Chest pain",
            "heart_rate": 82, "respiratory_rate": 16, "oxygen_saturation": 98,
            "pain_score": 2, "mental_status": "alert", "high_risk": False,
            "immediate_intervention": False, "expected_resources": 1,
        }
        model = create_triage_model([record], arrival_interval=3,
                                    treatment_duration=2, duration=8)
        model.run_for(1)
        self.assertEqual(model.patients[0].esi_level, 4)
        self.assertEqual(model.patients[0].symptoms, "Chest pain")
        self.assertEqual(model.patients[0].arrival_time, 0)
        self.assertIn("Assigned ESI Level 4", " ".join(model.patients[0].classification_trace))
        model.run_model()
        self.assertEqual(model.patients[0].status, "completed")

    def test_arrival_spacing_and_level_one_emergency_path(self):
        records = [
            {"patient_id": "M1-001", "age": 30, "expected_resources": 0,
             "immediate_intervention": True},
            {"patient_id": "M1-002", "age": 35, "expected_resources": 0},
        ]
        model = create_triage_model(records, arrival_interval=4,
                                    treatment_duration=2, duration=6)
        model.step()
        self.assertEqual(model.patients[0].esi_level, 1)
        self.assertEqual(model.patients[0].status, "emergency_treatment")
        model.run_for(4)
        self.assertEqual([p.arrival_time for p in model.patients], [0, 4])

    def test_member_one_clock_arrival_times_become_elapsed_minutes(self):
        records = [
            {"patient_id": "clock-1", "age": 30, "arrival_time": "09:05",
             "expected_resources": 0},
            {"patient_id": "clock-2", "age": 31, "arrival_time": "09:10",
             "expected_resources": 1},
        ]
        model = create_triage_model(records)
        model.run_for(6)
        self.assertEqual([p.arrival_time for p in model.patients], [0, 5])
        self.assertEqual(model.patients[1].esi_level, 4)

    def test_invalid_classification_is_rejected(self):
        with self.assertRaises(ValueError):
            create_triage_model([{"patient_id": "bad", "age": 12}]).step()


if __name__ == "__main__":
    unittest.main()
