"""Unit tests for agent fields, status transitions, and specialty rules."""

import unittest

from agents import DoctorAgent, PatientAgent
from model import EmergencyDepartmentModel


class AgentTests(unittest.TestCase):
    def test_patient_fields_and_status_transitions(self):
        model = EmergencyDepartmentModel(duration=5)
        patient = PatientAgent(model, "P100", 22, "Synthetic", 1, 3, 4,
                               required_specialty="Neurologist")
        patient.status = "waiting"
        patient.start_treatment(3, "D005")
        self.assertEqual(patient.waiting_time, 2)
        self.assertEqual(patient.display_status, "In Treatment")
        patient.complete_treatment(7)
        self.assertEqual(patient.treatment_end_time, 7)
        self.assertEqual(patient.display_status, "Completed")

    def test_doctor_counts_treatments_and_enforces_specialty(self):
        model = EmergencyDepartmentModel(duration=5)
        doctor = DoctorAgent(model, "Dcardio", "Cardiologist")
        patient = PatientAgent(model, "P1", 40, "Synthetic", 0, 2, 2,
                               required_specialty="Cardiologist")
        patient.status = "waiting"
        doctor.assign(patient, 0)
        self.assertEqual(doctor.availability_status, "Treating")
        self.assertEqual(doctor.available_at, 2)
        doctor.release(2)
        self.assertEqual(doctor.availability_status, "Available")
        self.assertEqual(doctor.patients_treated, 1)

    def test_specialty_values_are_required(self):
        model = EmergencyDepartmentModel(duration=2)
        with self.assertRaises(ValueError):
            PatientAgent(model, "P1", 20, "x", 0, 2, required_specialty=" ")


if __name__ == "__main__":
    unittest.main()
