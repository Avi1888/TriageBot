# ESI Triage Dataset – Data Dictionary

## 1. Dataset Overview

This dataset contains 50 synthetic adult patient records created for the educational TriageBot project.

The dataset is used to simulate emergency department triage and patient prioritization using the Emergency Severity Index (ESI) framework, Version 5.

Each record represents a simulated patient and contains demographic information, clinical observations, triage-related attributes, and simulation information.

The dataset is synthetic and does not contain real patient information. It is intended for educational and simulation purposes only and is not designed for real-world clinical decision-making.

## 2. Column Descriptions

| Column Name              | Description                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `patient_id`             | Unique identifier assigned to each synthetic patient.                                                        |
| `age`                    | Age of the patient in years. The dataset contains adult patients aged 18 and above.                          |
| `gender`                 | Gender of the synthetic patient.                                                                             |
| `arrival_time`           | Simulated time at which the patient arrives at the emergency department.                                     |
| `chief_complaint`        | Main symptom or reason for which the patient seeks emergency care.                                           |
| `heart_rate`             | Patient's heart rate measured in beats per minute (bpm).                                                     |
| `respiratory_rate`       | Number of breaths taken per minute.                                                                          |
| `oxygen_saturation`      | Percentage of oxygen saturation in the blood (SpO2).                                                         |
| `temperature`            | Body temperature in degrees Celsius.                                                                         |
| `pain_score`             | Reported pain intensity on a scale from 0 to 10.                                                             |
| `mental_status`          | Observed mental status of the patient, such as Alert, Confused, Lethargic, or Disoriented.                   |
| `high_risk`              | Boolean flag indicating whether the patient's condition is identified as high-risk for triage purposes.      |
| `required_specialty`     | Specialty assigned for simulation routing, such as General Physician, Surgeon, Cardiologist, or Neurologist. |
| `treatment_duration`     | Simulated duration of treatment in minutes.                                                                  |
| `immediate_intervention` | Boolean flag indicating whether the patient requires an immediate lifesaving intervention.                   |
| `expected_resources`     | Simplified estimate of the number of resource types expected during the patient's ED evaluation.             |

## 3. ESI Classification Logic

The triage classifier follows a simplified implementation of the Emergency Severity Index (ESI), Version 5.

### Decision 1 – Immediate Lifesaving Intervention

Patients requiring immediate lifesaving intervention are assigned ESI Level 1.

### Decision 2 – High-Risk Conditions

Patients flagged as high-risk or exhibiting altered mental status, such as confusion, lethargy, or disorientation, are assigned ESI Level 2.

Severe pain (pain score of 7 or above) is flagged for clinical assessment and consideration of Level 2, but is not automatically assigned Level 2 by the implemented classifier.

### Decision 3 – Expected Resources

For patients who do not meet the previous criteria, the simplified resource-based classification is:

| Expected Resource Types | ESI Level |
| ----------------------- | --------- |
| 2 or more               | Level 3   |
| 1                       | Level 4   |
| 0                       | Level 5   |

The dataset uses `expected_resources` values of 0, 1, and 2, where 2 represents two or more resource types.

### Decision 4 – Vital Signs Reassessment

The classifier flags the following adult vital signs for reassessment:

- Heart rate greater than 100 beats per minute
- Respiratory rate greater than 20 breaths per minute
- Oxygen saturation below 92%

Abnormal vital signs generate a warning in the classification trace. They do not automatically change the assigned ESI level.

## 4. Assumptions and Limitations

- The dataset contains only synthetic adult patients (18 years and above).
- Patient records are generated for educational and simulation purposes.
- The `high_risk` flag is predefined in the dataset and is used by the classifier.
- `expected_resources` is a simplified predefined estimate, not a prediction generated from a clinical assessment.
- Resource counts represent different resource types rather than the total number of individual tests or procedures.
- The classification logic is a simplified implementation of ESI Version 5 and does not cover every clinical scenario or nuance of the official algorithm.
- Abnormal vital signs generate reassessment warnings but do not automatically determine the triage level.
- The system does not replace trained healthcare professionals or clinical judgment.
- The prototype is not intended for use in real emergency departments or patient-care decisions.
