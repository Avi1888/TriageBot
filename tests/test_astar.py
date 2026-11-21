"""Optimality and admissibility checks for the finite-batch A* scheduler."""

import itertools
import unittest

from astar_scheduler import AStarScheduler, astar_schedule, brute_force_optimal_cost
from baseline_schedulers import compare_schedulers, esi_priority_schedule, fcfs_schedule


def p(pid, esi=3, arrival=0, specialty="General Physician", duration=2):
    return {"patient_id": pid, "age": 30, "arrival_time": arrival, "esi_level": esi,
            "required_specialty": specialty, "treatment_duration": duration}


def d(did, specialty="General Physician", available=0):
    return {"doctor_id": did, "specialty": specialty, "available_at": available}


class AStarTests(unittest.TestCase):
    def test_schedule_respects_specialty_and_returns_diagnostics(self):
        patients = [p("gp", esi=2), p("neuro", esi=3, specialty="Neurologist")]
        doctors = [d("D1"), d("D2", "Neurologist")]
        result = astar_schedule(patients, doctors)
        self.assertTrue(result["optimal"])
        self.assertTrue(result["complete"])
        self.assertGreaterEqual(result["expanded_states"], 1)
        self.assertGreaterEqual(result["execution_time_seconds"], 0)
        self.assertEqual({x["patient_id"]: x["doctor_id"] for x in result["schedule"]},
                         {"gp": "D1", "neuro": "D2"})

    def test_cost_calculation(self):
        patients = [p("urgent", esi=2, duration=2), p("low", esi=5, duration=1)]
        result = astar_schedule(patients, [d("D1")])
        self.assertEqual(result["weighted_wait_cost"], 2)
        self.assertEqual(sum(x["weighted_wait_cost"] for x in result["schedule"]), 2)

    def test_astar_matches_exhaustive_oracle(self):
        cases = [
            ([p("a", esi=2, duration=2), p("b", esi=5, duration=3)], [d("D1")]),
            ([p("a", esi=2, duration=3), p("b", esi=3, specialty="Surgeon", duration=2),
              p("c", esi=5, duration=1)], [d("D1"), d("D2", "Surgeon")]),
            ([p("a", esi=4, arrival=0), p("b", esi=2, arrival=2), p("c", esi=3, arrival=1)],
             [d("D1", available=2), d("D2", available=0)]),
        ]
        for patients, doctors in cases:
            with self.subTest(patient_count=len(patients)):
                exact = brute_force_optimal_cost(patients, doctors)
                result = astar_schedule(patients, doctors)
                self.assertEqual(result["weighted_wait_cost"], exact)
                self.assertTrue(result["optimal"])

    def test_proposed_heuristic_never_exceeds_exact_remaining_cost(self):
        patients = [p("g1", esi=2, duration=3), p("g2", esi=5, duration=2),
                    p("s1", esi=3, specialty="Surgeon", duration=4)]
        doctors = [d("GP", available=0), d("S", "Surgeon", 0)]
        scheduler = AStarScheduler()
        ids = [row["patient_id"] for row in patients]
        for mask in range(1, 1 << len(patients)):
            subset_ids = [ids[i] for i in range(len(ids)) if mask & (1 << i)]
            subset = [row for row in patients if row["patient_id"] in subset_ids]
            for gp_time, surgeon_time in itertools.product(range(4), repeat=2):
                availability = {"GP": gp_time, "S": surgeon_time}
                available_doctors = [d("GP", available=gp_time), d("S", "Surgeon", surgeon_time)]
                estimate = scheduler.estimate_remaining_cost(
                    patients, doctors, subset_ids, availability, current_time=0
                )
                exact_remaining = brute_force_optimal_cost(subset, available_doctors)
                self.assertLessEqual(estimate, exact_remaining)

    def test_zero_heuristic_returns_same_optimal_cost(self):
        patients = [p("a", esi=2, duration=3), p("b", esi=4, arrival=1, duration=2),
                    p("c", esi=5, duration=2)]
        doctors = [d("D1"), d("D2")]
        informed = astar_schedule(patients, doctors, use_heuristic=True)
        zero = astar_schedule(patients, doctors, use_heuristic=False)
        self.assertEqual(informed["weighted_wait_cost"], zero["weighted_wait_cost"])
        self.assertEqual(informed["weighted_wait_cost"], brute_force_optimal_cost(patients, doctors))

    def test_tie_breaking_is_deterministic(self):
        patients = [p("b", esi=3), p("a", esi=3)]
        doctors = [d("D2"), d("D1")]
        one = astar_schedule(patients, doctors)
        two = astar_schedule(patients, doctors)
        self.assertEqual(one["schedule"], two["schedule"])

    def test_missing_specialty_is_not_assigned_to_wrong_doctor(self):
        result = astar_schedule([p("n", specialty="Neurologist")], [d("D1")])
        self.assertEqual(result["schedule"], [])
        self.assertEqual(result["unschedulable"][0]["patient_id"], "n")
        self.assertIn("Neurologist", result["unschedulable"][0]["reason"])

    def test_esi_one_is_rejected_from_ordinary_astar(self):
        with self.assertRaises(ValueError):
            astar_schedule([p("critical", esi=1)], [d("D1")])

    def test_baseline_order_and_shared_eligibility_rules(self):
        patients = [p("old", esi=5, arrival=0, duration=2),
                    p("urgent", esi=2, arrival=1, duration=2)]
        doctors = [d("D1")]
        fcfs = fcfs_schedule(patients, doctors)
        priority = esi_priority_schedule(patients, doctors)
        self.assertEqual(fcfs["schedule"][0]["patient_id"], "old")
        self.assertEqual(priority["schedule"][0]["patient_id"], "urgent")
        for result in (fcfs, priority):
            self.assertTrue(all(row["doctor_id"] == "D1" for row in result["schedule"]))

    def test_all_three_schedulers_use_same_inputs(self):
        patients = [p("a", esi=2, duration=3), p("b", esi=5, duration=2)]
        roster = [d("GP1"), d("GP2")]
        results = compare_schedulers(patients, roster)
        self.assertEqual(set(results), {"astar", "fcfs", "esi_priority"})
        for result in results.values():
            self.assertEqual({x["patient_id"] for x in result["schedule"]}, {"a", "b"})
        self.assertLessEqual(results["astar"]["weighted_wait_cost"],
                             results["fcfs"]["weighted_wait_cost"])

    def test_search_limit_is_reported_as_non_optimal(self):
        rows = [p(f"P{i}", esi=2) for i in range(5)]
        result = astar_schedule(rows, [d("D1")], max_expansions=1)
        self.assertFalse(result["optimal"])
        self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
