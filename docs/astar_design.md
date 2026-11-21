# A* scheduler design

## Why A*

The scheduling objective has multiple competing constraints: patient urgency,
arrival time, doctor availability, treatment durations, and fixed doctor
specialties. A* explores different legal patient-to-doctor assignments and
finds a minimum-cost ordering for a small, bounded waiting batch. FCFS and ESI
priority baselines provide simpler comparisons.

ESI weights are experimental objective weights (ESI 2: 4, ESI 3: 3, ESI 4: 2,
ESI 5: 1), not clinical values. ESI 1 is rejected by the ordinary scheduler and
handled by the separate model pathway.

## Search state and actions

The canonical state key contains:

1. The sorted tuple of remaining patient indexes.
2. A tuple of the next available minute for each doctor, kept in stable doctor
   ID order.

Each queue node also carries its partial schedule so the winning plan can be
returned. The path does not affect future costs once the remaining patients and
doctor availability are known, so it is intentionally omitted from the state
key. Equal-specialty doctors at the same availability are symmetric; only one
of those equivalent actions is expanded.

An action selects any unscheduled patient and a doctor with exactly matching
`required_specialty`. If no doctor in the roster covers a patient's specialty,
that patient is returned as unschedulable and is excluded from the feasible
search batch.

## Transition and cost

For patient `i` and eligible doctor `d`:

```text
start_i = max(current_minute, arrival_i, doctor_available_at_d)
wait_i  = max(0, start_i - arrival_i)
end_i   = start_i + treatment_duration_i
```

The selected doctor's next-available time becomes `end_i`, and the patient is
removed from the remaining tuple.

The path cost is:

```text
g(n) = sum(weight[ESI_i] * wait_i) for patients already scheduled
f(n) = g(n) + h(n)
```

The implementation uses `heapq`, a monotonic counter for deterministic tie
ordering, and a best-known g-score for each canonical state. If a state is
reached with a lower cost, it is queued again. The first goal removed from the
queue is returned when search completes under the documented assumptions.

## Heuristic and admissibility assumptions

For every remaining patient `i`, the heuristic independently selects the
earliest start offered by any doctor with matching specialty, using that
doctor's current next-available time:

```text
h(n) = sum(weight[ESI_i] * max(0, earliest_eligible_start_i - arrival_i))
```

This ignores competition among remaining patients. Any feasible continuation
must assign each patient to one of those same eligible doctors, and assigning
another remaining patient can only keep or delay a doctor's availability. Thus
each independent earliest start is no later than that patient's start in a
joint feasible schedule. With non-negative weights, summing these optimistic
wait costs is a lower bound on the remaining schedule cost.

This reasoning assumes a static batch, fixed specialties, one non-preemptive
slot per doctor, deterministic treatment durations, non-negative urgency
weights, and no external events changing the doctor roster while that search is
running. Dynamic events cause a new search rather than changing the active
search state. Admissibility is checked by tests against exhaustive enumeration
for small instances; no consistency claim is made. Passing
`use_heuristic=False` runs the same search with `h(n)=0` and is also compared
with the exhaustive oracle in tests.

## Dynamic replanning in Mesa

The model builds a fresh batch from eligible waiting patients after arrivals
and at treatment-completion boundaries. A doctor's current treatment remains
fixed. Its `available_at` value gives A* the future time when that slot opens.
The model commits only assignments whose planned start is the current minute;
patients assigned to a future slot remain waiting and are included in the next
replan. An incoming patient can therefore change the future plan without
interrupting a treatment in progress.

## Limits

A* is exponential in the number of jobs. The model defaults to batches of eight
and caps a search at 250,000 expanded states. A result that reaches the cap
reports `optimal=False` and `complete=False`; it must not be described as
optimal. Only one feasible batch is optimized at a time, so the model is a
replanning simulation and not a global all-day schedule optimizer. The
synthetic urgency weights and treatment times are not clinical guidance.

## Validation

`tests/test_astar.py` compares A* with exhaustive enumeration, checks the
heuristic against true remaining costs for small states, compares informed
search with zero-heuristic search, verifies specialty constraints, and checks
that an expansion-limited result reports its non-optimal status accurately.
