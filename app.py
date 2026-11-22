import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from triage_rules import classify_patient
from model import EmergencyDepartmentModel, SCENARIOS
from agents import SPECIALTIES

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="TriageBot | Emergency Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------
# CUSTOM STYLING (fixes dark-mode metric text)
# -----------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
    }

    [data-testid="stMetricLabel"] p {
        color: #4b5563 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] div {
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

PATIENT_COLUMNS = [
    "patient_id",
    "name",
    "age",
    "symptoms",
    "required_specialty",
    "heart_rate",
    "respiratory_rate",
    "oxygen_saturation",
    "immediate_intervention",
    "high_risk",
    "mental_status",
    "pain_score",
    "expected_resources",
    "arrival_time",
    "treatment_duration",
    "esi_level",
    "reason",
    "decision_trace",
    "status",
]


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    return str(val).strip().lower() in {"true", "1", "yes", "y"}


def _count_resources(val) -> int:
    if pd.isna(val) or val is None:
        return 0
    if isinstance(val, (int, float)):
        return max(0, int(val))
    text = str(val).strip()
    if not text or text.lower() in {"none", "nan", "0"}:
        return 0
    if text.isdigit():
        return int(text)
    # Handle pipe- or comma-separated resource categories (e.g., "laboratory|imaging")
    parts = [p.strip() for p in text.replace(",", "|").split("|") if p.strip()]
    return len(set(parts))


def normalize_and_classify_record(raw: dict, default_arrival: int = 0) -> dict:
    """Normalize CSV or manual fields to match Member 1 & Member 2 contracts."""
    # Support both snake_case (Member 1/2) and Title Case (earlier Member 3 CSVs)
    lower_map = {str(k).strip().lower(): v for k, v in raw.items()}

    def pick(*keys, default=None):
        for k in keys:
            if k in lower_map and pd.notna(lower_map[k]) and str(lower_map[k]).strip() != "":
                return lower_map[k]
        return default

    patient_id = str(pick("patient_id", "patient id", default="")).strip()
    name = str(pick("name", default=patient_id or "Unknown")).strip()
    age = int(float(pick("age", default=25)))
    symptoms = str(pick("symptoms", "chief_complaint", default="General symptoms")).strip()
    specialty = str(pick("required_specialty", "specialty", default="General Physician")).strip()
    if specialty not in SPECIALTIES:
        specialty = "General Physician"

    heart_rate = int(float(pick("heart_rate", "heart rate", default=80)))
    respiratory_rate = int(float(pick("respiratory_rate", "respiratory rate", default=16)))
    oxygen_saturation = int(float(pick("oxygen_saturation", "spo2", default=98)))

    immediate_intervention = _to_bool(pick("immediate_intervention", default=False))
    high_risk = _to_bool(pick("high_risk", default=False))

    mental_status = str(pick("mental_status", default="")).strip().lower()
    if not mental_status:
        mental_status = "confused" if _to_bool(pick("altered_mental_status", default=False)) else "normal"

    pain_score = int(float(pick("pain_score", default=8 if _to_bool(pick("severe_pain_distress", default=False)) else 2)))

    raw_resources = pick("expected_resources", "resources", default=0)
    expected_resources = _count_resources(raw_resources)

    arrival_time = int(float(pick("arrival_time", "arrival time", default=default_arrival)))
    treatment_duration = int(float(pick("treatment_duration", default=15)))

    patient_dict = {
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "symptoms": symptoms,
        "required_specialty": specialty,
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "oxygen_saturation": oxygen_saturation,
        "immediate_intervention": immediate_intervention,
        "high_risk": high_risk,
        "mental_status": mental_status,
        "pain_score": pain_score,
        "expected_resources": expected_resources,
        "arrival_time": arrival_time,
        "treatment_duration": treatment_duration,
    }

    result = classify_patient(patient_dict)
    patient_dict["esi_level"] = result.get("level")
    patient_dict["reason"] = result.get("reason", "")
    patient_dict["decision_trace"] = " → ".join(result.get("trace", []))
    patient_dict["status"] = "Waiting" if result.get("level") is not None else "Unassessed"
    return patient_dict


def _clock_times_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert HH:MM arrival times (as in data/patients.csv) to minutes from the first arrival."""
    df = df.copy()
    col = next((c for c in df.columns if str(c).strip().lower() in {"arrival_time", "arrival time"}), None)
    if col is None or not df[col].astype(str).str.contains(":").any():
        return df

    def to_min(v):
        h, m = str(v).strip().split(":")[:2]
        return int(h) * 60 + int(m)

    mins = df[col].map(to_min)
    df[col] = (mins - mins.min()).astype(int)
    return df


def load_records_into_state(df: pd.DataFrame) -> tuple[int, list[str]]:
    """Classify rows from a DataFrame and append non-duplicate valid patients."""
    df = _clock_times_to_minutes(df)
    existing_ids = set(st.session_state.patients["patient_id"].astype(str))
    added_rows = []
    errors = []

    for idx, row in df.iterrows():
        rec = normalize_and_classify_record(row.to_dict(), default_arrival=idx * 2)
        if not rec["patient_id"]:
            rec["patient_id"] = f"P{len(existing_ids) + len(added_rows) + 1:03d}"
        if rec["patient_id"] in existing_ids:
            continue
        if rec["esi_level"] is None:
            errors.append(f"{rec['patient_id']}: {rec['reason']}")
            continue
        existing_ids.add(rec["patient_id"])
        added_rows.append(rec)

    if added_rows:
        st.session_state.patients = pd.concat(
            [st.session_state.patients, pd.DataFrame(added_rows)],
            ignore_index=True,
        )
    return len(added_rows), errors


# -----------------------------------------
# SESSION STATE
# -----------------------------------------

if "patients" not in st.session_state:
    st.session_state.patients = pd.DataFrame(columns=PATIENT_COLUMNS)

if "sim_model" not in st.session_state:
    st.session_state.sim_model = None

if "sim_metrics_df" not in st.session_state:
    st.session_state.sim_metrics_df = pd.DataFrame()

if "sim_patient_results_df" not in st.session_state:
    st.session_state.sim_patient_results_df = pd.DataFrame()

if "sim_wait_by_esi_df" not in st.session_state:
    st.session_state.sim_wait_by_esi_df = pd.DataFrame()

# -----------------------------------------
# SIDEBAR
# -----------------------------------------

st.sidebar.title("🏥 TriageBot")
st.sidebar.caption("Emergency Department Management")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Patient Registration",
        "Patient Queue",
        "Simulation",
        "Analytics",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "TriageBot is an educational ESI rule-based and Mesa agent-based "
    "emergency triage simulation prototype. Not for clinical use."
)

# -----------------------------------------
# DASHBOARD
# -----------------------------------------

if page == "Dashboard":
    st.title("🏥 Emergency Department Dashboard")
    st.write("Welcome to TriageBot — ESI Rule-Based Triage & Mesa Agent Simulation.")

    model = st.session_state.sim_model
    patients = st.session_state.patients

    if model is not None:
        statuses = pd.DataFrame(model.get_patient_statuses())
        roster = pd.DataFrame(model.get_doctor_roster())
        total = len(statuses)
        waiting = int((statuses["status"] == "Waiting").sum()) if total else 0
        treated = int((statuses["status"] == "Treated").sum()) if total else 0
        avail_docs = int(roster["available"].sum()) if not roster.empty else 0
        total_docs = len(roster)
        doc_display = f"{avail_docs} / {total_docs}"
    else:
        total = len(patients)
        waiting = int((patients["status"] == "Waiting").sum()) if total else 0
        treated = int((patients["status"] == "Treated").sum()) if total else 0
        doc_display = "Ready (Run Sim)"

    st.subheader("Hospital Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", total)
    c2.metric("Waiting Patients", waiting)
    c3.metric("Treated Patients", treated)
    c4.metric("Doctors Available", doc_display)

    st.divider()

    if model is not None:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader(f"Simulation Patient States (Minute {model.current_minute} / {model.duration})")
            st.dataframe(pd.DataFrame(model.get_patient_statuses()), use_container_width=True, hide_index=True)
        with col_right:
            st.subheader("Doctor Roster")
            st.dataframe(
                pd.DataFrame(model.get_doctor_roster())[["doctor_id", "specialty", "availability_status", "current_patient"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.subheader("Registered & Classified Patient Records")
        if total > 0:
            display_cols = ["patient_id", "name", "age", "symptoms", "required_specialty", "esi_level", "reason", "status"]
            st.dataframe(patients[display_cols].tail(15), use_container_width=True, hide_index=True)
        else:
            st.info("No patients registered yet. Go to Patient Registration to add patients or load data/patients.csv.")

# -----------------------------------------
# PATIENT REGISTRATION
# -----------------------------------------

elif page == "Patient Registration":
    st.title("📝 Patient Registration & ESI Assessment")
    st.write("Assess adult patients (18+) using Member 1's ESI rule engine or load synthetic CSV scenarios.")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            patient_id = st.text_input("Patient ID", value=f"P{len(st.session_state.patients) + 1:03d}")
            name = st.text_input("Patient Name")
            age = st.number_input("Age (18+)", min_value=0, max_value=120, value=35)
            required_specialty = st.selectbox("Required Specialty", list(SPECIALTIES))
            arrival_time = st.number_input("Arrival Minute (for Simulation)", min_value=0, max_value=480, value=0)

        with col2:
            symptoms = st.text_area("Symptoms / Chief Complaint", height=105)
            heart_rate = st.number_input("Heart Rate (BPM)", min_value=20, max_value=250, value=82)
            respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=16)
            oxygen_saturation = st.number_input("SpO2 (%)", min_value=50, max_value=100, value=98)

        with col3:
            immediate_intervention = st.checkbox("Requires Immediate Life-Saving Intervention (Decision 1)")
            high_risk = st.checkbox("High-Risk Presentation (Decision 2)")
            mental_status = st.selectbox("Mental Status", ["normal", "confused", "lethargic", "disoriented"])
            pain_score = st.slider("Pain Score (0–10)", min_value=0, max_value=10, value=2)
            expected_resources = st.number_input(
                "Expected Distinct Resource Types (0, 1, or 2+)",
                min_value=0,
                max_value=10,
                value=1,
            )

        submitted = st.form_submit_button("Assess & Register Patient", use_container_width=True)

        if submitted:
            if not patient_id.strip():
                st.error("Please enter a Patient ID.")
            elif patient_id.strip() in st.session_state.patients["patient_id"].astype(str).values:
                st.error("This Patient ID already exists.")
            else:
                record = normalize_and_classify_record({
                    "patient_id": patient_id.strip(),
                    "name": name.strip() or patient_id.strip(),
                    "age": age,
                    "symptoms": symptoms.strip() or "Unspecified",
                    "required_specialty": required_specialty,
                    "heart_rate": heart_rate,
                    "respiratory_rate": respiratory_rate,
                    "oxygen_saturation": oxygen_saturation,
                    "immediate_intervention": immediate_intervention,
                    "high_risk": high_risk,
                    "mental_status": mental_status,
                    "pain_score": pain_score,
                    "expected_resources": expected_resources,
                    "arrival_time": arrival_time,
                })

                if record["esi_level"] is None:
                    st.error(f"Patient Rejected by ESI Rules: {record['reason']}")
                    st.write("**Decision Trace:**", record["decision_trace"])
                else:
                    st.session_state.patients = pd.concat(
                        [st.session_state.patients, pd.DataFrame([record])],
                        ignore_index=True,
                    )
                    st.success(f"Patient {record['patient_id']} classified as **ESI Level {record['esi_level']}** — {record['reason']}")
                    st.info(f"**Decision Trace:** {record['decision_trace']}")

    st.divider()

    st.subheader("Load Synthetic Dataset or Upload CSV")

    col_btn, col_up = st.columns([1, 2])
    with col_btn:
        default_csv = os.path.join("data", "patients.csv")
        if os.path.exists(default_csv):
            if st.button("📂 Load Project Dataset (data/patients.csv)", use_container_width=True):
                try:
                    df_default = pd.read_csv(default_csv)
                    added, errs = load_records_into_state(df_default)
                    st.success(f"Loaded and classified {added} patients from `data/patients.csv`.")
                    if errs:
                        st.warning(f"Skipped {len(errs)} unsupported/invalid rows (e.g. age < 18).")
                except Exception as exc:
                    st.error(f"Failed to load data/patients.csv: {exc}")

    with col_up:
        uploaded_file = st.file_uploader("Or upload a custom CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.dataframe(uploaded_df.head(), use_container_width=True)
                if st.button("Import & Classify Uploaded CSV"):
                    added, errs = load_records_into_state(uploaded_df)
                    st.success(f"Imported and classified {added} new patients.")
                    if errs:
                        st.warning(f"Skipped {len(errs)} rows: {errs[:5]}")
            except Exception as error:
                st.error(f"Unable to read CSV: {error}")

# -----------------------------------------
# PATIENT QUEUE
# -----------------------------------------

elif page == "Patient Queue":
    st.title("👨‍⚕️ Priority Patient Queue")

    model = st.session_state.sim_model
    patients = st.session_state.patients

    if model is not None:
        st.subheader(f"Live Mesa Simulation Queue (Minute {model.current_minute})")
        queue_agents = model.get_patient_queue()
        if queue_agents:
            q_df = pd.DataFrame([
                {
                    "Patient ID": p.patient_id,
                    "ESI Level": p.esi_level,
                    "Symptoms": p.symptoms,
                    "Required Specialty": p.required_specialty,
                    "Arrival Minute": p.arrival_time,
                    "Waiting Time (min)": p.waiting_time,
                    "Status": p.display_status,
                }
                for p in queue_agents
            ])
            st.dataframe(q_df, use_container_width=True, hide_index=True)
        else:
            st.info("No patients currently waiting in the active simulation queue.")

        st.divider()
        st.subheader("All Simulation Patients")
        st.dataframe(pd.DataFrame(model.get_patient_statuses()), use_container_width=True, hide_index=True)

    elif len(patients) > 0:
        st.subheader("Pre-Simulation Priority Queue (Sorted by ESI Level, then Arrival Time)")
        sorted_df = patients.sort_values(by=["esi_level", "arrival_time", "patient_id"])
        display_cols = ["patient_id", "name", "esi_level", "reason", "required_specialty", "arrival_time", "status"]
        st.dataframe(sorted_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No patients in the queue yet. Register patients or load `data/patients.csv` first.")

# -----------------------------------------
# SIMULATION
# -----------------------------------------

elif page == "Simulation":
    st.title("⚙️ Mesa Emergency Department Simulation")
    st.write("Run Member 2's agent-based Mesa simulation with specialty-aware A* scheduling.")

    col1, col2 = st.columns(2)

    with col1:
        doctors = st.slider("Number of Doctors", min_value=1, max_value=15, value=5)
        simulation_time = st.slider("Simulation Duration (minutes)", min_value=30, max_value=480, value=120, step=30)

    with col2:
        scenario_label = st.selectbox(
            "Select Scenario",
            ["normal", "surge", "limited_staffing"],
            format_func=lambda s: {
                "normal": "Normal Operations (normal)",
                "surge": "Emergency Surge (surge)",
                "limited_staffing": "Limited Staffing (limited_staffing)",
            }[s],
        )

        patient_source = st.radio(
            "Patient Arrival Source",
            ["Use Registered / CSV Patients", "Generate Synthetic Arrivals"],
            index=0 if len(st.session_state.patients) > 0 else 1,
        )

    seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42)

    c_run, c_step, c_reset = st.columns(3)

    def build_model() -> EmergencyDepartmentModel:
        if patient_source == "Use Registered / CSV Patients":
            if len(st.session_state.patients) == 0:
                st.warning("No registered patients found—falling back to synthetic arrival generation.")
                return EmergencyDepartmentModel(
                    n_doctors=doctors,
                    duration=simulation_time,
                    scenario=scenario_label,
                    classifier=classify_patient,
                    seed=int(seed),
                )
            records = st.session_state.patients.to_dict(orient="records")
            return EmergencyDepartmentModel(
                patients=records,
                n_doctors=doctors,
                duration=simulation_time,
                scenario=scenario_label,
                classifier=classify_patient,
                seed=int(seed),
            )
        return EmergencyDepartmentModel(
            n_doctors=doctors,
            duration=simulation_time,
            scenario=scenario_label,
            classifier=classify_patient,
            seed=int(seed),
        )

    def sync_model_outputs(model: EmergencyDepartmentModel):
        st.session_state.sim_model = model
        st.session_state.sim_metrics_df = model.get_metrics()
        st.session_state.sim_patient_results_df = model.get_patient_results()
        st.session_state.sim_wait_by_esi_df = model.get_waiting_time_by_esi()

    if c_run.button("▶ Run Full Simulation", use_container_width=True):
        model = build_model()
        model.run_model()
        sync_model_outputs(model)
        st.success(f"Completed {model.current_minute}-minute simulation ({scenario_label}).")

    if c_step.button("⏭ Step +10 Minutes", use_container_width=True):
        model = st.session_state.sim_model or build_model()
        model.run_for(10)
        sync_model_outputs(model)
        st.info(f"Advanced simulation to minute {model.current_minute} / {model.duration}.")

    if c_reset.button("🔄 Reset Simulation", use_container_width=True):
        st.session_state.sim_model = None
        st.session_state.sim_metrics_df = pd.DataFrame()
        st.session_state.sim_patient_results_df = pd.DataFrame()
        st.session_state.sim_wait_by_esi_df = pd.DataFrame()
        st.info("Simulation state reset.")

    model = st.session_state.sim_model
    if model is not None:
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        statuses_df = pd.DataFrame(model.get_patient_statuses())
        m1.metric("Simulated Minute", f"{model.current_minute} / {model.duration}")
        m2.metric("Patients Treated", int((statuses_df["status"] == "Treated").sum()) if not statuses_df.empty else 0)
        m3.metric("Max Queue Length", model.maximum_queue_length)
        m4.metric("Doctor Utilization", f"{model.doctor_utilisation * 100:.1f}%")

        st.subheader("Doctor Roster & Utilization")
        st.dataframe(pd.DataFrame(model.get_doctor_roster()), use_container_width=True, hide_index=True)

        st.subheader("Patient Simulation Outcomes")
        st.dataframe(st.session_state.sim_patient_results_df, use_container_width=True, hide_index=True)

# -----------------------------------------
# ANALYTICS
# -----------------------------------------

elif page == "Analytics":
    st.title("📊 Simulation & Triage Analytics")

    metrics_df = st.session_state.sim_metrics_df
    results_df = st.session_state.sim_patient_results_df
    wait_esi_df = st.session_state.sim_wait_by_esi_df
    patients = st.session_state.patients

    if not metrics_df.empty and not results_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patients by ESI Level")
            esi_counts = results_df["esi_level"].value_counts().sort_index()
            fig1, ax1 = plt.subplots()
            esi_counts.plot(kind="bar", ax=ax1, color="#1d4ed8")
            ax1.set_xlabel("ESI Level (1 = Immediate, 5 = Non-Urgent)")
            ax1.set_ylabel("Number of Patients")
            ax1.set_title("ESI Acuity Distribution")
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.subheader("Waiting Queue Length Over Time")
            fig2, ax2 = plt.subplots()
            time_col = "minute" if "minute" in metrics_df.columns else metrics_df.columns[0]
            queue_col = "queue_length" if "queue_length" in metrics_df.columns else metrics_df.columns[1]
            ax2.plot(metrics_df[time_col], metrics_df[queue_col], color="#dc2626", linewidth=2)
            ax2.set_xlabel("Simulated Minute")
            ax2.set_ylabel("Waiting Queue Length")
            ax2.set_title("Queue Evolution Over Simulated Time")
            st.pyplot(fig2)
            plt.close(fig2)

        st.divider()

        st.subheader("Average & Median Waiting Time by ESI Level")
        if not wait_esi_df.empty:
            st.dataframe(wait_esi_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Export Simulation Results")
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                label="⬇ Download Patient Results (CSV)",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="patient_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_exp2:
            st.download_button(
                label="⬇ Download Time-Series Metrics (CSV)",
                data=metrics_df.to_csv(index=False).encode("utf-8"),
                file_name="simulation_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

    elif len(patients) > 0:
        st.info("Showing pre-simulation ESI breakdown. Run a simulation on the Simulation page for queue & wait-time charts.")
        esi_counts = patients["esi_level"].value_counts().sort_index()
        fig, ax = plt.subplots()
        esi_counts.plot(kind="bar", ax=ax, color="#1d4ed8")
        ax.set_xlabel("ESI Level")
        ax.set_ylabel("Patients")
        ax.set_title("Registered Patient ESI Distribution")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Register patients or run a simulation first to view analytics and export CSVs.")

# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()
st.caption(
    "TriageBot | Integrated Prototype (ESI Rules + Mesa Simulation + Streamlit Dashboard) | "
    "Educational simulation only. Not for clinical use."
)
