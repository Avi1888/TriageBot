def classify_patient(patient):
    """
    Classify a patient using the ESI framework.
    Returns ESI level, reason, and decision trace.
    """

    trace = []

    if not isinstance(patient, dict):
        return {
            "level": None,
            "reason": "Invalid patient data.",
            "trace": ["Patient data must be provided as a dictionary."]
        }

    if "age" not in patient:
        return {
            "level": None,
            "reason": "Missing patient age.",
            "trace": ["Age information is required."]
        }

    try:
        age = int(patient["age"])
    except (ValueError, TypeError):
        return {
            "level": None,
            "reason": "Invalid patient age.",
            "trace": ["Age must be a valid number."]
        }

    if age < 18:
        return {
            "level": None,
            "reason": "Pediatric patients are outside the scope of this prototype.",
            "trace": ["This prototype is designed for adult patients (18+)."]
        }

    # Decision 1: Does the patient require an immediate life-saving intervention?
    immediate_intervention = patient.get("immediate_intervention", False)

    trace.append("Decision 1: Check for immediate life-saving intervention.")

    if immediate_intervention:
        return {
            "level": 1,
            "reason": "Requires immediate life-saving intervention.",
            "trace": trace
        }

    trace.append("No immediate life-saving intervention required.")

    # Decision 2: Is the patient high-risk?
    trace.append("Decision 2: Check for high-risk presentation.")

    if patient.get("high_risk", False):
        return {
            "level": 2,
            "reason": "High-risk presentation.",
            "trace": trace
        }

    mental_status = patient.get("mental_status", "").lower()

    if mental_status in ["confused", "lethargic", "disoriented"]:
        return {
            "level": 2,
            "reason": "Altered mental status.",
            "trace": trace
        }

    if patient.get("pain_score", 0) >= 7:
        trace.append("Severe pain reported; requires clinical assessment for ESI Level 2.")

    trace.append("No Level 2 criteria identified.")

        # Decision 3: Determine the expected number of resource types
    trace.append("Decision 3: Determine expected resource types.")

    expected_resources = patient.get("expected_resources")

    if expected_resources is None:
        return {
            "level": None,
            "reason": "Expected resource information is missing.",
            "trace": trace
        }

    try:
        expected_resources = int(expected_resources)
    except (ValueError, TypeError):
        return {
            "level": None,
            "reason": "Invalid expected resource information.",
            "trace": trace
        }

    if expected_resources < 0:
        return {
            "level": None,
            "reason": "Expected resources cannot be negative.",
            "trace": trace
        }

    if expected_resources >= 2:
        level = 3
        reason = "Two or more expected resource types."
    elif expected_resources == 1:
        level = 4
        reason = "One expected resource type."
    else:
        level = 5
        reason = "No expected resource types."

    trace.append(f"Expected resource types: {expected_resources}.")
    trace.append(f"Assigned ESI Level {level} based on expected resources.")

    # Decision 4: Reassess adult vital signs
    trace.append("Decision 4: Reassess abnormal adult vital signs.")

    heart_rate = patient.get("heart_rate")
    respiratory_rate = patient.get("respiratory_rate")
    oxygen_saturation = patient.get("oxygen_saturation")

    abnormal_vitals = []

    if heart_rate is not None and heart_rate > 100:
        abnormal_vitals.append("heart rate above 100")

    if respiratory_rate is not None and respiratory_rate > 20:
        abnormal_vitals.append("respiratory rate above 20")

    if oxygen_saturation is not None and oxygen_saturation < 92:
        abnormal_vitals.append("oxygen saturation below 92%")

    if abnormal_vitals:
        trace.append(
            "Abnormal vital signs detected: "
            + ", ".join(abnormal_vitals)
            + ". Clinical reassessment is required."
        )
    else:
        trace.append("No abnormal vital signs detected using the implemented thresholds.")

    return {
        "level": level,
        "reason": reason,
        "trace": trace
    }