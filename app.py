# app.py
# Streamlit app: PivotPath — AI Career Advisor (clean, robust, role-aware)
# Place this file in the same folder as your model/encoder files (optional).
# Run: streamlit run app.py

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="PivotPath — AI Career Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Helper: file-safe loading
# -------------------------
def safe_load_pickle(path):
    import pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def safe_load_pickle(path):
    import joblib
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -------------------------
# Attempt to load assets (optional)
# -------------------------
ASSET_DIR = Path(".")
salary_model = safe_load_pickle(ASSET_DIR / "salary_model.pkl")
career_model = safe_load_pickle(ASSET_DIR / "career_recommender.pkl")
scaler = safe_load_pickle(ASSET_DIR / "scaler.pkl") or safe_load_joblib(ASSET_DIR / "scaler.pkl")
encoders = {}
for name in [
    "current_job_title_encoder.pkl",
    "target_job_title_encoder.pkl",
    "education_level_encoder.pkl",
    "portfolio_strength_encoder.pkl",
    "certifications_encoder.pkl",
    "language_proficiency_encoder.pkl",
    "remote_work_preference_encoder.pkl",
]:
    enc = safe_load_joblib(ASSET_DIR / name)
    if enc is not None:
        encoders[name.replace("_encoder.pkl", "")] = enc

feat_salary = load_json(ASSET_DIR / "feature_columns_salary.json")
feat_career = load_json(ASSET_DIR / "feature_columns_career.json")

# -------------------------
# Role — static dictionaries (customize)
# -------------------------
# Use these role mappings to create role-specific skill gaps and courses.
ROLE_SKILL_GAPS = {
    "Data Scientist": [
        "Python (Pandas, NumPy)",
        "SQL & Data querying",
        "Machine Learning fundamentals",
        "Data visualization (Matplotlib/Seaborn)",
        "Model deployment (Flask/Streamlit)"
    ],
    "Frontend Engineer": [
        "JavaScript (ES6+)",
        "React fundamentals",
        "HTML/CSS responsive design",
        "Web performance & accessibility",
        "Frontend testing (Jest)"
    ],
    "System Administrator": [
        "Linux fundamentals",
        "Shell scripting",
        "Networking basics",
        "Cloud fundamentals (AWS/GCP)",
        "Monitoring & troubleshooting"
    ],
    "UI/UX Designer": [
        "Design thinking & user research",
        "Figma/Sketch/Adobe XD",
        "Wireframing & prototyping",
        "Interaction design",
        "Storytelling & presentation"
    ],
    "Product Manager": [
        "User research & product discovery",
        "Roadmapping & prioritization",
        "Stakeholder communication",
        "Metrics & experimentation",
        "Basic data analysis"
    ],
    "Junior Data Analyst": [
        "Excel & spreadsheets",
        "SQL basics",
        "Data cleaning & EDA",
        "Basic visualization",
        "Presentation skills"
    ],
    "Frontend Developer": [
        "HTML/CSS/JS",
        "React or Vue",
        "Component architecture",
        "State management",
        "Testing"
    ],
    "Technical Support Engineer": [
        "Troubleshooting fundamentals",
        "Customer communication",
        "Product knowledge",
        "Basic networking",
        "Scripting for automation"
    ],
    "Junior UI/UX Designer": [
        "Figma basics",
        "Wireframing",
        "Basic prototyping",
        "Usability heuristics",
        "Design critique"
    ],
    "Marketing Analyst": [
        "Excel & Google Sheets",
        "Basic statistics",
        "Data visualization",
        "Marketing analytics tools",
        "Campaign measurement"
    ],
}

ROLE_COURSES = {
    "Data Scientist": [
        {"platform": "Coursera", "course": "Python for Data Science", "duration": "4 weeks"},
        {"platform": "Udemy", "course": "Machine Learning Bootcamp", "duration": "6 weeks"},
        {"platform": "YouTube", "course": "Data Viz with Tableau", "duration": "3 weeks"},
    ],
    "UI/UX Designer": [
        {"platform": "Coursera", "course": "UI/UX Specialization", "duration": "6 weeks"},
        {"platform": "YouTube", "course": "Design system tutorials", "duration": "2 weeks"},
        {"platform": "Udemy", "course": "Figma practical projects", "duration": "4 weeks"},
    ],
    # ... default fallback
}
# make sure every role has a courses entry (fallback to Data Scientist)
for r in ROLE_SKILL_GAPS.keys():
    ROLE_COURSES.setdefault(r, ROLE_COURSES.get(r, ROLE_COURSES.get("Data Scientist")))

# -------------------------
# UI: Sidebar — inputs (blank by default)
# -------------------------
st.sidebar.title("Tell us about yourself")
st.sidebar.caption("Start by selecting the fields below, then click Analyze.")

# First option value is a blank 'Select...' so user must choose
current_job_options = ["Select current job"] + list(ROLE_SKILL_GAPS.keys())
target_job_options = ["Select target job"] + list(ROLE_SKILL_GAPS.keys())
education_options = ["Select education"] + [
    "Associate Degree in IT",
    "Bachelor's in IT",
    "Bachelor's in Computer Science",
    "Bachelor's in Design",
    "Bachelor's in Marketing",
    "MBA",
    "Bachelor's in Statistics",
]

current_job = st.sidebar.selectbox("Current Job", current_job_options, index=0)
target_job = st.sidebar.selectbox("Target Job", target_job_options, index=0)
education = st.sidebar.selectbox("Education level", education_options, index=0)

portfolio_strength = st.sidebar.selectbox(
    "Portfolio Strength", ["Select...", "Low", "Medium", "High"], index=0
)
certifications = st.sidebar.selectbox(
    "Certifications", ["Select...", "None", "AWS Certified", "Certified UX Designer", "DataCamp Track"], index=0
)
language_proficiency = st.sidebar.selectbox(
    "Language Proficiency", ["Select...", "Beginner", "Intermediate", "Advanced"], index=0
)
remote_pref = st.sidebar.selectbox(
    "Remote Preference", ["Select...", "Onsite", "Hybrid", "Remote"], index=0
)

years_exp = st.sidebar.number_input("Years of Experience", min_value=0, max_value=40, value=0, step=1)
projects_completed = st.sidebar.number_input("Projects Completed", min_value=0, max_value=50, value=0, step=1)
comm_rating = st.sidebar.slider("Communication Rating (1-10)", min_value=1, max_value=10, value=5)
skills_text = st.sidebar.text_input("Skills (comma-separated)", placeholder="e.g. Python, SQL, Tableau")

analyze = st.sidebar.button("Get Career Analysis")

# Quick status (files loaded)
status_container = st.sidebar.expander("Files status (debug)", expanded=False)
with status_container:
    st.write("salary_model:", "FOUND" if salary_model else "NOT FOUND")
    st.write("career_model:", "FOUND" if career_model else "NOT FOUND")
    st.write("scaler:", "FOUND" if scaler else "NOT FOUND")
    st.write("feature_columns_salary:", "FOUND" if feat_salary else "NOT FOUND")
    st.write("feature_columns_career:", "FOUND" if feat_career else "NOT FOUND")
    st.write("encoders loaded:", list(encoders.keys()) or "NONE")

# -------------------------
# Main area: header
# -------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("## 🚀 PivotPath — AI Career Advisor")
    st.write("Professional, friendly career guidance — role-aware skill gaps, curated courses and a story-like roadmap.")
with header_col2:
    st.markdown("")  # reserved for future widgets

st.divider()

# -------------------------
# If user didn't click analyze, show clean blanks
# -------------------------
if not analyze:
    st.info("Fill the inputs on the left and click **Get Career Analysis** to generate recommendations.")
    st.stop()  # nothing else runs until user clicks analyze

# -------------------------
# Validate inputs
# -------------------------
errors = []
if current_job == "Select current job":
    errors.append("Choose your current job.")
if target_job == "Select target job":
    errors.append("Choose your target job.")
if education == "Select education":
    errors.append("Select your education level.")
if portfolio_strength == "Select...":
    errors.append("Select portfolio strength.")
if certifications == "Select...":
    certifications = "None"  # accept None
if language_proficiency == "Select...":
    errors.append("Select language proficiency.")
if remote_pref == "Select...":
    errors.append("Select remote preference.")

if errors:
    st.error("Please fix these inputs:")
    for e in errors:
        st.write("- " + e)
    st.stop()

# -------------------------
# Build profile summary
# -------------------------
profile_col, quickstatus_col = st.columns([3, 1])
with profile_col:
    st.markdown("### 👤 Profile Summary")
    st.write(f"**Current:** {current_job}  →  **Target:** {target_job}")
    st.write(f"**Experience:** {years_exp} years • **Projects:** {projects_completed}")
    st.write(f"**Skills:** {skills_text or '—'}  •  **Skill count:** {len([s for s in skills_text.split(',') if s.strip()])}")
    st.write(f"**Portfolio:** {portfolio_strength}  •  **Certs:** {certifications}")

with quickstatus_col:
    st.markdown("### 🔧 Quick Status")
    st.write("- Salary model: " + ("✅" if salary_model else "❌"))
    st.write("- Career model: " + ("✅" if career_model else "❌"))
    st.write("- Scaler: " + ("✅" if scaler else "❌"))
    st.write("- Feature cols (salary): " + ("✅" if feat_salary else "❌"))
    st.write("- Feature cols (career): " + ("✅" if feat_career else "❌"))

st.divider()

# -------------------------
# Salary prediction (INR only)
# -------------------------
salary_box = st.container()
with salary_box:
    st.markdown("### 🪙 Estimated Salary (INR per year)")

    # Try model prediction; otherwise fallback.
    predicted_salary_inr = None
    if salary_model and feat_salary:
        try:
            # Build input dict using feature column names if available
            # If encoders exist, transform categorical features accordingly; otherwise use naive mapping.
            X = {}
            for feat in feat_salary:
                # Some features may not be provided; fill with defaults
                if feat in ["estimated_salary_usd"]:  # drop target if present
                    continue
                val = None
                if feat == "current_job_title":
                    # If we have encoder for current_job_title
                    enc = encoders.get("current_job_title")
                    if enc:
                        try:
                            val = int(enc.transform([current_job])[0])
                        except Exception:
                            val = 0
                    else:
                        val = 0
                elif feat == "target_job_title":
                    enc = encoders.get("target_job_title")
                    if enc:
                        try:
                            val = int(enc.transform([target_job])[0])
                        except Exception:
                            val = 0
                    else:
                        val = 0
                elif feat == "education_level":
                    enc = encoders.get("education_level")
                    if enc:
                        try:
                            val = int(enc.transform([education])[0])
                        except Exception:
                            val = 0
                    else:
                        val = 0
                elif feat == "portfolio_strength":
                    mapping = {"Low": 0, "Medium": 1, "High": 2}
                    val = mapping.get(portfolio_strength, 1)
                elif feat == "certifications":
                    # count or map certificate
                    val = 0 if certifications in (None, "None") else 1
                elif feat == "language_proficiency":
                    mapping = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
                    val = mapping.get(language_proficiency, 1)
                elif feat == "remote_work_preference":
                    mapping = {"Onsite": 0, "Hybrid": 1, "Remote": 2}
                    val = mapping.get(remote_pref, 0)
                elif feat in ["years_of_experience", "projects_completed", "communication_skills_rating"]:
                    if feat == "years_of_experience":
                        val = years_exp
                    elif feat == "projects_completed":
                        val = projects_completed
                    else:
                        val = comm_rating
                else:
                    # other engineered features
                    val = 0
                X[feat] = val

            X_df = pd.DataFrame([X])
            # scale if scaler exists
            if scaler:
                try:
                    # Only scale the intersection of numeric features/scaler
                    X_df_scaled = X_df.copy()
                    # Scaler expects same order/columns; attempt transform but catch failure
                    X_df_scaled[X_df.columns] = scaler.transform(X_df[X_df.columns])
                    X_df = X_df_scaled
                except Exception:
                    pass

            salary_usd = float(salary_model.predict(X_df)[0])
            # convert USD to INR using a conservative fixed multiplier (no external call)
            USD_TO_INR = 82.0
            predicted_salary_inr = int(salary_usd * USD_TO_INR)
        except Exception:
            predicted_salary_inr = None

    # Fallback heuristic: role-based base salaries (INR)
    if predicted_salary_inr is None:
        BASE_BY_ROLE_INR = {
            "Data Scientist": 900000,
            "Frontend Engineer": 700000,
            "System Administrator": 500000,
            "UI/UX Designer": 600000,
            "Product Manager": 1000000,
            "Junior Data Analyst": 350000,
            "Frontend Developer": 650000,
            "Technical Support Engineer": 300000,
            "Junior UI/UX Designer": 350000,
            "Marketing Analyst": 350000,
        }
        predicted_salary_inr = BASE_BY_ROLE_INR.get(target_job, 450000)
        # adjust roughly by experience
        predicted_salary_inr = int(predicted_salary_inr * (1 + min(years_exp, 10) * 0.06))

    # Show salary card
    st.success(f"₹ {predicted_salary_inr:,} INR per year")

st.divider()

# -------------------------
# Career suggestion (respect user target)
# -------------------------
st.markdown("### 🎯 Best Career Suggestion")
st.info(f"**Recommended role (user-chosen):** {target_job}")

# Note: If you want to also display what the model _would_ predict (optional), show below
if career_model and feat_career:
    try:
        # create a simple input based on feat_career (best-effort)
        Xin = {}
        for feat in feat_career:
            if feat == "target_job_title":
                Xin[feat] = 0
            elif feat == "current_job_title":
                enc = encoders.get("current_job_title")
                Xin[feat] = int(enc.transform([current_job])[0]) if enc else 0
            elif feat == "years_of_experience":
                Xin[feat] = years_exp
            else:
                Xin[feat] = 0
        Xin_df = pd.DataFrame([Xin])
        if scaler:
            try:
                Xin_df[Xin_df.columns] = scaler.transform(Xin_df[Xin_df.columns])
            except Exception:
                pass
        pred_label = career_model.predict(Xin_df)[0]
        # If we have a target job encoder, inverse transform to a string
        enc_target = encoders.get("target_job_title")
        model_pred_role = None
        if enc_target:
            try:
                # invert label (if classifier used numeric labels)
                model_pred_role = enc_target.inverse_transform([int(pred_label)])[0]
            except Exception:
                model_pred_role = str(pred_label)
        else:
            model_pred_role = str(pred_label)
        st.write("Model's suggestion (if available):", model_pred_role)
    except Exception:
        st.write("Model's suggestion: —")
else:
    st.write("Model's suggestion: — (model not available)")

st.divider()

# -------------------------
# Skill gap and recommended courses (role-aware)
# -------------------------
st.markdown("### 🧠 Skill Gap — Improve These (role-aware)")
gaps = ROLE_SKILL_GAPS.get(target_job, ROLE_SKILL_GAPS["Data Scientist"])
for g in gaps:
    st.write(f"- {g}")

st.markdown("### 📚 Suggested Courses")
courses = ROLE_COURSES.get(target_job, ROLE_COURSES["Data Scientist"])
courses_df = pd.DataFrame(courses)
st.table(courses_df)

st.divider()

# -------------------------
# Roadmap — Story-like (Role-Specific)
# -------------------------

ROLE_ROADMAP = {
    "Data Scientist": {
        "Assess": "Review Python, SQL, portfolio, machine learning basics.",
        "Learn": "Complete ML courses (Supervised, Unsupervised), Stats, EDA.",
        "Build": "Create 4 ML projects — Regression, Classification, NLP, Deployment.",
        "Profile": "Publish Kaggle notebooks, GitHub repos & LinkedIn posts.",
        "Apply": "Apply to DS roles, practice ML system design interviews."
    },
    "Frontend Engineer": {
        "Assess": "Evaluate HTML, CSS, JS, React fundamentals.",
        "Learn": "Master React, Hooks, State mgmt, API integration.",
        "Build": "Build 3–5 responsive websites + React apps.",
        "Profile": "Deploy on Netlify/Vercel, share UI case studies.",
        "Apply": "Apply with live websites + GitHub portfolio."
    },
    "System Administrator": {
        "Assess": "Check Linux knowledge, shell scripting, networking basics.",
        "Learn": "Study Linux, Bash, AWS, monitoring tools.",
        "Build": "Build server configs, automate tasks, deploy scripts.",
        "Profile": "Document troubleshooting cases, publish scripts on GitHub.",
        "Apply": "Apply to SysAdmin/IT roles & take tech support interviews."
    },
    "UI/UX Designer": {
        "Assess": "Evaluate Figma, wireframing, user research skills.",
        "Learn": "Practice UX case studies, interaction design, heuristics.",
        "Build": "Create 3 UI/UX case studies + mobile/web prototypes.",
        "Profile": "Publish Dribbble/Figma/Behance portfolio.",
        "Apply": "Apply for UI/UX Intern/Junior roles with strong portfolio."
    },
    "Product Manager": {
        "Assess": "Check communication, research, analytics & docs skills.",
        "Learn": "Study PRDs, user stories, Agile, prioritization frameworks.",
        "Build": "Create 2 product case studies + a product roadmap.",
        "Profile": "Publish PM case studies & feature breakdowns.",
        "Apply": "Apply for APM roles, complete PM interview practice."
    },
    "Junior Data Analyst": {
        "Assess": "Assess Excel, SQL, dashboarding skills.",
        "Learn": "Take courses in SQL, Power BI/Tableau, statistics.",
        "Build": "Create 3 dashboards + 2 data cleaning projects.",
        "Profile": "Upload dashboards to LinkedIn & GitHub.",
        "Apply": "Apply for Data Analyst Intern/Junior roles."
    },
    "Frontend Developer": {
        "Assess": "Check React, HTML, JS, responsiveness knowledge.",
        "Learn": "React advanced concepts + CSS frameworks (Tailwind).",
        "Build": "Build 4–6 UI clones & dynamic web apps.",
        "Profile": "Deploy projects, share UI/UX improvements.",
        "Apply": "Apply with portfolio & GitHub repos."
    },
    "Technical Support Engineer": {
        "Assess": "Assess troubleshooting, communication & software basics.",
        "Learn": "Study OS basics, networking, ticket systems.",
        "Build": "Create documentation tutorials & automation scripts.",
        "Profile": "Show real scenarios & solutions in resume.",
        "Apply": "Apply to L1 support roles, practice customer interaction."
    },
    "Junior UI/UX Designer": {
        "Assess": "Check Figma basics, wireframing, UX research understanding.",
        "Learn": "Learn design principles, layouts, typography.",
        "Build": "Make 3 beginner-friendly app/web redesigns.",
        "Profile": "Publish Behance/Figma case studies.",
        "Apply": "Apply for entry UI/UX roles."
    },
    "Marketing Analyst": {
        "Assess": "Check analytics tools, Excel, reports knowledge.",
        "Learn": "Study Google Analytics, basic statistics, dashboards.",
        "Build": "Create 3–5 campaign analysis projects.",
        "Profile": "Publish dashboard snapshots & insights.",
        "Apply": "Apply for junior marketing/analytics roles."
    }
}

# Select roadmap based on target job
roadmap = ROLE_ROADMAP.get(target_job, ROLE_ROADMAP["Data Scientist"])

st.markdown("### 🗺️ Roadmap — Personalized for Your Role")

rc1, rc2, rc3, rc4, rc5 = st.columns([1,1,1,1,1])

with rc1:
    st.subheader("🔍 Assess")
    st.write(roadmap["Assess"])

with rc2:
    st.subheader("📘 Learn")
    st.write(roadmap["Learn"])

with rc3:
    st.subheader("🛠 Build")
    st.write(roadmap["Build"])

with rc4:
    st.subheader("📣 Profile")
    st.write(roadmap["Profile"])

with rc5:
    st.subheader("🚀 Apply")
    st.write(roadmap["Apply"])

st.success("Your personalized role-based roadmap is ready!")

# -------------------------
# Footer
# -------------------------
st.markdown("Made with ❤️ — PivotPath")
