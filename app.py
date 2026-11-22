
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="TriageBot | Emergency Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# CUSTOM STYLING
# -----------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    h1, h2, h3 {
        color: #17365d;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# SESSION STATE
# -----------------------------------------

if "patients" not in st.session_state:
    st.session_state.patients = pd.DataFrame(
        columns=[
            "Patient ID",
            "Name",
            "Age",
            "Gender",
            "Symptoms",
            "Heart Rate",
            "Temperature",
            "ESI Level",
            "Status",
            "Arrival Time"
        ]
    )

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
        "Analytics"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "TriageBot is an educational emergency "
    "triage simulation project."
)

# -----------------------------------------
# DASHBOARD
# -----------------------------------------

if page == "Dashboard":

    st.title("🏥 Emergency Department Dashboard")
    st.write(
        "Welcome to TriageBot — Patient Triage "
        "and Emergency Management System."
    )

    patients = st.session_state.patients

    total = len(patients)
    waiting = (
        patients["Status"] == "Waiting"
    ).sum() if total else 0

    treated = (
        patients["Status"] == "Treated"
    ).sum() if total else 0

    st.subheader("Hospital Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Patients", total)
    c2.metric("Waiting Patients", waiting)
    c3.metric("Treated Patients", treated)
    c4.metric("Doctors Available", "—")

    st.divider()

    st.subheader("Recent Patient Records")

    if total > 0:
        st.dataframe(
            patients.tail(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "No patients registered yet. "
            "Go to Patient Registration to add patients."
        )

    st.caption(
        "Doctor availability and simulation metrics "
        "will be connected to the simulation module."
    )

# -----------------------------------------
# PATIENT REGISTRATION
# -----------------------------------------

elif page == "Patient Registration":

    st.title("📝 Patient Registration")

    st.write(
        "Enter patient details for emergency assessment."
    )

    with st.form("patient_form"):

        col1, col2 = st.columns(2)

        with col1:
            patient_id = st.text_input("Patient ID")
            name = st.text_input("Patient Name")
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=25
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Other", "Prefer not to say"]
            )

        with col2:
            symptoms = st.text_area(
                "Symptoms / Chief Complaint"
            )

            heart_rate = st.number_input(
                "Heart Rate (BPM)",
                min_value=0,
                max_value=300,
                value=80
            )

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=25.0,
                max_value=45.0,
                value=37.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Register Patient",
            use_container_width=True
        )

        if submitted:

            if not patient_id.strip() or not name.strip():
                st.error("Please enter Patient ID and Name.")

            elif patient_id in st.session_state.patients[
                "Patient ID"
            ].values:
                st.error("This Patient ID already exists.")

            else:
                new_patient = {
                    "Patient ID": patient_id.strip(),
                    "Name": name.strip(),
                    "Age": age,
                    "Gender": gender,
                    "Symptoms": symptoms,
                    "Heart Rate": heart_rate,
                    "Temperature": temperature,
                    "ESI Level": "Pending",
                    "Status": "Waiting",
                    "Arrival Time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }

                st.session_state.patients = pd.concat(
                    [
                        st.session_state.patients,
                        pd.DataFrame([new_patient])
                    ],
                    ignore_index=True
                )

                st.success(
                    f"Patient {name} registered successfully!"
                )

                st.info(
                    "ESI classification is pending integration "
                    "with Member 1's triage rules."
                )

    st.divider()

    st.subheader("Upload Patient CSV")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_data = pd.read_csv(uploaded_file)

            st.write("CSV Preview")
            st.dataframe(
                uploaded_data.head(),
                use_container_width=True
            )

            required_columns = [
                "Patient ID",
                "Name",
                "Age",
                "Gender",
                "Symptoms"
            ]

            missing = [
                col for col in required_columns
                if col not in uploaded_data.columns
            ]

            if missing:
                st.error(
                    f"Missing required columns: {missing}"
                )

            elif st.button("Import Patient Records"):

                existing_ids = set(
                    st.session_state.patients["Patient ID"]
                )

                if uploaded_data["Patient ID"].isna().any():
                    st.error("Patient IDs cannot be empty.")

                elif uploaded_data["Patient ID"].astype(
                    str
                ).duplicated().any():
                    st.error("Duplicate IDs found in uploaded CSV.")

                else:
                    uploaded_data["Patient ID"] = (
                        uploaded_data["Patient ID"].astype(str)
                    )

                    new_data = uploaded_data[
                        ~uploaded_data["Patient ID"].isin(existing_ids)
                    ].copy()

                    for col, default in {
                        "Heart Rate": 0,
                        "Temperature": 0.0,
                        "ESI Level": "Pending",
                        "Status": "Waiting",
                        "Arrival Time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    }.items():
                        if col not in new_data.columns:
                            new_data[col] = default

                    st.session_state.patients = pd.concat(
                        [
                            st.session_state.patients,
                            new_data
                        ],
                        ignore_index=True
                    )

                    st.success(
                        f"{len(new_data)} new patients imported."
                    )

        except Exception as error:
            st.error(f"Unable to read CSV: {error}")

# -----------------------------------------
# PATIENT QUEUE
# -----------------------------------------

elif page == "Patient Queue":

    st.title("👨‍⚕️ Patient Queue Management")

    patients = st.session_state.patients

    if len(patients) == 0:
        st.info("No patients in the queue yet.")

    else:
        st.subheader("Current Patient Queue")

        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Waiting", "In Treatment", "Treated"]
        )

        if status_filter == "All":
            filtered = patients
        else:
            filtered = patients[
                patients["Status"] == status_filter
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Update Patient Status")

        selected_id = st.selectbox(
            "Select Patient",
            patients["Patient ID"].tolist()
        )

        new_status = st.selectbox(
            "New Status",
            ["Waiting", "In Treatment", "Treated"]
        )

        if st.button("Update Status"):
            st.session_state.patients.loc[
                st.session_state.patients["Patient ID"]
                == selected_id,
                "Status"
            ] = new_status

            st.success("Patient status updated.")
            st.rerun()

# -----------------------------------------
# SIMULATION
# -----------------------------------------

elif page == "Simulation":

    st.title("⚙️ Emergency Department Simulation")

    st.write(
        "Configure simulation parameters for testing "
        "patient flow and hospital capacity."
    )

    col1, col2 = st.columns(2)

    with col1:
        doctors = st.slider(
            "Number of Doctors",
            min_value=1,
            max_value=20,
            value=5
        )

        simulation_time = st.slider(
            "Simulation Duration (minutes)",
            min_value=30,
            max_value=480,
            value=120,
            step=30
        )

    with col2:
        scenario = st.selectbox(
            "Select Scenario",
            [
                "Normal Patient Flow",
                "Emergency Surge",
                "Limited Staff"
            ]
        )

        patient_count = st.number_input(
            "Number of Simulated Patients",
            min_value=1,
            max_value=500,
            value=50
        )

    st.divider()

    st.subheader("Simulation Configuration")

    st.write(f"Doctors: {doctors}")
    st.write(f"Duration: {simulation_time} minutes")
    st.write(f"Scenario: {scenario}")
    st.write(f"Patients: {patient_count}")

    if st.button("▶ Run Simulation", use_container_width=True):

        st.warning(
            "Simulation engine is not connected yet. "
            "Member 2's Mesa model will be integrated here."
        )

# -----------------------------------------
# ANALYTICS
# -----------------------------------------

elif page == "Analytics":

    st.title("📊 Patient Analytics")

    patients = st.session_state.patients

    if len(patients) == 0:

        st.info(
            "Register or import patients to view analytics."
        )

    else:

        st.subheader("Patient Status Distribution")

        status_counts = patients["Status"].value_counts()

        fig, ax = plt.subplots()

        status_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Patient Status")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Patient Queue Overview")

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        st.subheader("Age Distribution")

        fig2, ax2 = plt.subplots()

        patients["Age"].plot(
            kind="hist",
            bins=10,
            ax=ax2
        )

        ax2.set_xlabel("Age")
        ax2.set_ylabel("Number of Patients")
        ax2.set_title("Patient Age Distribution")

        st.pyplot(fig2)
        plt.close(fig2)

        st.divider()

        st.subheader("Export Patient Data")

        csv_data = patients.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Patient Records",
            data=csv_data,
            file_name="triagebot_patient_records.csv",
            mime="text/csv"
        )

# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "TriageBot | Member 3 - Dashboard Development | "
    "Educational simulation only. Not for clinical use."
)
