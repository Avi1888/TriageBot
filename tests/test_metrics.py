"""Validation of model-level and patient-level Pandas metrics."""

import unittest

from metrics import SimulationMetrics
from model import EmergencyDepartmentModel


class MetricsTests(unittest.TestCase):
    def test_metrics_values_and_dataframes(self):
        patients = [
            {"patient_id": "A", "age": 40, "arrival_time": 0, "esi_level": 2,
             "required_specialty": "General Physician", "treatment_duration": 1},
            {"patient_id": "B", "age": 50, "arrival_time": 0, "esi_level": 5,
             "required_specialty": "General Physician", "treatment_duration": 1},
        ]
        roster = [("D1", "General Physician")]
        model = EmergencyDepartmentModel(patients=patients, doctor_roster=roster,
                                         duration=5, batch_size=1)
        model.run_model()
        frame = model.get_metrics()
        results = model.get_patient_results()
        self.assertEqual(frame.iloc[-1]["number_completed"], 2)
        self.assertEqual(frame.iloc[-1]["patient_throughput"], 2)
        self.assertEqual(frame.iloc[-1]["maximum_patient_waiting_time"], 1)
        self.assertEqual(list(results.columns[:6]), ["patient_id", "age", "symptoms", "arrival_time", "esi_level", "required_specialty"])
        self.assertGreaterEqual(frame.iloc[-1]["doctor_utilisation"], 0)
        by_esi = model.get_waiting_time_by_esi()
        self.assertEqual(by_esi.loc[by_esi.esi_level == 2, "average_waiting_time"].iloc[0], 0)
        self.assertEqual(by_esi.loc[by_esi.esi_level == 5, "average_waiting_time"].iloc[0], 1)

    def test_empty_wait_summary_returns_zero_rows_by_level(self):
        result = SimulationMetrics.waiting_time_by_esi([])
        self.assertEqual(len(result), 5)
        self.assertTrue((result.average_waiting_time == 0).all())


if __name__ == "__main__":
    unittest.main()
