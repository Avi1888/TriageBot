import unittest
from data_loader import load_patients
import pandas as pd
import tempfile

class TestDataLoader(unittest.TestCase):

    def test_load_patients(self):
        patients = load_patients()

        self.assertEqual(len(patients), 50)
        self.assertIsInstance(patients, list)
        self.assertIsInstance(patients[0], dict)

    def test_patient_has_required_fields(self):
        patients = load_patients()

        required_fields = [
            "patient_id",
            "age",
            "chief_complaint",
            "expected_resources"
        ]

        for patient in patients:
            for field in required_fields:
                self.assertIn(field, patient)

    def test_unique_patient_ids(self):
        patients = load_patients()

        patient_ids = [
            patient["patient_id"] for patient in patients
        ]

        self.assertEqual(len(patient_ids), len(set(patient_ids)))

    def test_invalid_age(self):

        patients = pd.read_csv("data/patients.csv")
        patients.loc[0, "age"] = 15

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            patients.to_csv(temp.name, index=False)

            with self.assertRaises(ValueError):
                load_patients(temp.name)
    def test_invalid_oxygen_saturation(self):

        patients = pd.read_csv("data/patients.csv")
        patients.loc[0, "oxygen_saturation"] = 150

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            patients.to_csv(temp.name, index=False)

            with self.assertRaises(ValueError):
                load_patients(temp.name)

    def test_duplicate_patient_ids(self):

        patients = pd.read_csv("data/patients.csv")

        # Give the second patient the same ID as the first
        patients.loc[1, "patient_id"] = patients.loc[0, "patient_id"]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            patients.to_csv(temp.name, index=False)

            with self.assertRaises(ValueError):
                load_patients(temp.name)

    def test_missing_column(self):

        patients = pd.read_csv("data/patients.csv")

        # Remove the age column
        patients = patients.drop(columns=["age"])

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            patients.to_csv(temp.name, index=False)

            with self.assertRaises(ValueError):
                load_patients(temp.name)

if __name__ == "__main__":
    unittest.main()