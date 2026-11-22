"""Prototype rule-based ESI classification supplied by Member 1.

This educational implementation is included here to make the integration
branch runnable. It is not clinically validated and must not guide real care.
"""


def classify_patient(patient):
    """Return a prototype ESI level, explanation, and decision trace."""
    trace = []
    if not isinstance(patient, dict):
        return {"level": None, "reason": "Invalid patient data.",
                "trace": ["Patient data must be provided as a dictionary."]}
    if "age" not in patient:
        return {"level": None, "reason": "Missing patient age.",
                "trace": ["Age information is required."]}
    try:
        age = int(patient["age"])
    except (ValueError, TypeError):
        return {"level": None, "reason": "Invalid patient age.",
                "trace": ["Age must be a valid number."]}
    if age < 18:
        return {"level": None,
                "reason": "Pediatric patients are outside the scope of this prototype.",
                "trace": ["This prototype is designed for adult patients (18+)."]}

    trace.append("Decision 1: Check for immediate life-saving intervention.")
    if patient.get("immediate_intervention", False):
        return {"level": 1, "reason": "Requires immediate life-saving intervention.", "trace": trace}
    trace.append("No immediate life-saving intervention required.")

    trace.append("Decision 2: Check for high-risk presentation.")
    if patient.get("high_risk", False):
        return {"level": 2, "reason": "High-risk presentation.", "trace": trace}
    mental_status = patient.get("mental_status", "")
    if isinstance(mental_status, str) and mental_status.lower() in {
        "confused", "lethargic", "disoriented"
    }:
        return {"level": 2, "reason": "Altered mental status.", "trace": trace}
    if patient.get("pain_score", 0) >= 7:
        trace.append("Severe pain reported; requires clinical assessment for ESI Level 2.")
    trace.append("No Level 2 criteria identified.")

    # Fixed the extra indentation in Member 1's branch, which prevented import.
    trace.append("Decision 3: Determine expected resource types.")
    expected_resources = patient.get("expected_resources")
    if expected_resources is None:
        return {"level": None, "reason": "Expected resource information is missing.",
                "trace": trace}
    try:
        expected_resources = int(expected_resources)
    except (ValueError, TypeError):
        return {"level": None, "reason": "Invalid expected resource information.",
                "trace": trace}
    if expected_resources < 0:
        return {"level": None, "reason": "Expected resources cannot be negative.",
                "trace": trace}

    if expected_resources >= 2:
        level, reason = 3, "Two or more expected resource types."
    elif expected_resources == 1:
        level, reason = 4, "One expected resource type."
    else:
        level, reason = 5, "No expected resource types."
    trace.extend([f"Expected resource types: {expected_resources}.",
                  f"Assigned ESI Level {level} based on expected resources."])

    trace.append("Decision 4: Reassess adult vital signs.")
    abnormal_vitals = []
    heart_rate = patient.get("heart_rate")
    respiratory_rate = patient.get("respiratory_rate")
    oxygen_saturation = patient.get("oxygen_saturation")
    if heart_rate is not None and heart_rate > 100:
        abnormal_vitals.append("heart rate above 100")
    if respiratory_rate is not None and respiratory_rate > 20:
        abnormal_vitals.append("respiratory rate above 20")
    if oxygen_saturation is not None and oxygen_saturation < 92:
        abnormal_vitals.append("oxygen saturation below 92%")
    if abnormal_vitals:
        trace.append("Abnormal vital signs detected: " + ", ".join(abnormal_vitals)
                     + ". Clinical reassessment is required.")
    else:
        trace.append("No abnormal vital signs detected using the implemented thresholds.")
    return {"level": level, "reason": reason, "trace": trace}
