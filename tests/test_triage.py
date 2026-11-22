import unittest
from triage_rules import classify_patient


class TestTriage(unittest.TestCase):

    def test_invalid_patient_input(self):
        result = classify_patient(None)
        self.assertIsNone(result["level"])
        self.assertEqual(result["reason"], "Invalid patient data.")

    def test_missing_age(self):
        result = classify_patient({})
        self.assertIsNone(result["level"])
        self.assertEqual(result["reason"], "Missing patient age.")

    def test_invalid_age(self):
        result = classify_patient({"age": "abc"})
        self.assertIsNone(result["level"])
        self.assertEqual(result["reason"], "Invalid patient age.")

    def test_pediatric_patient(self):
        result = classify_patient({"age": 15})
        self.assertIsNone(result["level"])

    def test_level_1(self):
        result = classify_patient({
            "age": 25,
            "immediate_intervention": True
        })
        self.assertEqual(result["level"], 1)

    def test_adult_without_immediate_intervention(self):
        result = classify_patient({
            "age": 25,
            "immediate_intervention": False
        })
        self.assertIsNone(result["level"])

    def test_level_2_high_risk(self):
        result = classify_patient({
            "age": 25,
            "immediate_intervention": False,
            "high_risk": True
        })
        self.assertEqual(result["level"], 2)
        self.assertEqual(result["reason"], "High-risk presentation.")


    def test_level_2_altered_mental_status(self):
        result = classify_patient({
            "age": 25,
            "immediate_intervention": False,
            "high_risk": False,
            "mental_status": "confused"
        })
        self.assertEqual(result["level"], 2)
        self.assertEqual(result["reason"], "Altered mental status.")

    def test_level_3(self):
        result = classify_patient({
            "age": 25,
            "immediate_intervention": False,
            "high_risk": False,
            "mental_status": "Alert",
            "pain_score": 3,
            "expected_resources": 2
        })
        self.assertEqual(result["level"], 3)

    def test_level_4(self):
        result = classify_patient({
            "age": 25,
            "immediate_intervention": False,
            "high_risk": False,
            "mental_status": "Alert",
            "pain_score": 3,
            "expected_resources": 1
        })
        self.assertEqual(result["level"], 4)

    def test_level_5(self):
        result = classify_patient({
            "age": 25,
            "immediate_intervention": False,
            "high_risk": False,
            "mental_status": "Alert",
            "pain_score": 3,
            "expected_resources": 0
        })
        self.assertEqual(result["level"], 5)

    def test_abnormal_vitals_in_trace(self):
        result = classify_patient({
            "age": 25,
            "immediate_intervention": False,
            "high_risk": False,
            "mental_status": "Alert",
            "pain_score": 3,
            "expected_resources": 1,
            "heart_rate": 110,
            "respiratory_rate": 24,
            "oxygen_saturation": 90
        })

        self.assertEqual(result["level"], 4)
        self.assertTrue(
            any("Abnormal vital signs detected" in item
                for item in result["trace"])
        )

if __name__ == "__main__":
    unittest.main()
