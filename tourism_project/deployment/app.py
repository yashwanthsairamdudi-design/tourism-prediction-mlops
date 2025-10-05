import os
import io
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download, HfApi
# --- be compatible with multiple hub versions ---
try:
    from huggingface_hub.utils import HfHubHTTPError  # most versions
except Exception:  # fallback for very old versions
    class HfHubHTTPError(Exception):
        pass

# ---------------------------
# Page & Environment
# ---------------------------
st.set_page_config(page_title="Visit With Us — Wellness Package Predictor", page_icon="🧭", layout="centered")

# Configure via Space / environment variables (Settings → Variables & secrets)
MODEL_REPO = os.getenv("MODEL_REPO", "Yash0204/tourism-prediction-mlops")   # HF model repo: owner/name
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best_tourism_wellness_model_v1.joblib")  # exact file in that repo
REPO_TYPE = os.getenv("MODEL_REPO_TYPE", "model")  # usually "model"
CLASSIFICATION_THRESHOLD = float(os.getenv("THRESHOLD", "0.45"))

# ---------------------------
# Model loader (lazy, cached)
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model(repo_id: str, filename: str, repo_type: str = "model"):
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        return joblib.load(path)
    except Exception as e:
        # Show what files actually exist to help diagnose 404s
        files = None
        try:
            files = [f.rfilename for f in HfApi().list_repo_files(repo_id, repo_type=repo_type)]
        except Exception:
            pass
        st.error(
            "Failed to load model from Hugging Face Hub.\n\n"
            f"- repo_id: `{repo_id}` (type={repo_type})\n"
            f"- filename requested: `{filename}`\n"
            f"- files available: {files if files is not None else '[could not list]'}\n\n"
            f"Original error: {e}"
        )
        raise

# ---------------------------
# UI
# ---------------------------
st.title("🧭 Visit With Us — Wellness Tourism Purchase Prediction")
st.write("Predict whether a customer is likely to purchase the **Wellness Tourism Package**.")

with st.expander("Prediction options", expanded=False):
    threshold = st.slider("Classification threshold", 0.05, 0.95, CLASSIFICATION_THRESHOLD, 0.01)
    st.caption("Scores ≥ threshold → **Positive (Will Purchase)**; otherwise **Negative**.")

# ---- Inputs matching the dataset schema ----
st.subheader("Customer Details")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", 0, 120, 35)
    typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    citytier = st.selectbox("City Tier", [1, 2, 3], index=0)
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with c2:
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_persons = st.number_input("Number Of Person Visiting", 1, 20, 2)
    pref_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5], index=2)
    num_trips = st.number_input("Number Of Trips (per year)", 0, 100, 2)
    num_children = st.number_input("Number Of Children Visiting (<5y)", 0, 10, 0)

st.subheader("Documents & Assets")
c3, c4 = st.columns(2)
with c3:
    passport = st.selectbox("Has Passport", [0, 1], index=1)
    owncar = st.selectbox("Owns Car", [0, 1], index=0)
    designation = st.selectbox("Designation", ["Junior", "Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"], index=2)
with c4:
    monthly_income = st.number_input("Monthly Income", 0, 10_000_000, 60_000, 1_000)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Queen", "Other"])
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 4)

st.subheader("Sales Interaction")
c5, c6 = st.columns(2)
with c5:
    num_followups = st.number_input("Number Of Followups", 0, 50, 2)
with c6:
    duration_pitch = st.number_input("Duration Of Pitch (minutes)", 0, 300, 20)

single_input = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "MaritalStatus": marital,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": pref_star,
    "NumberOfTrips": num_trips,
    "NumberOfChildrenVisiting": num_children,
    "Passport": int(passport),
    "OwnCar": int(owncar),
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch
}])

st.markdown("#### Preview")
st.dataframe(single_input, use_container_width=True)

# ---------------------------
# Load model and helpers
# ---------------------------
with st.spinner("Loading model from Hugging Face…"):
    model = load_model(MODEL_REPO, MODEL_FILENAME, REPO_TYPE)
    st.success(f"Model loaded: **{MODEL_REPO} / {MODEL_FILENAME}**")

def predict_df(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    proba = model.predict_proba(df)[:, 1]
    label = (proba >= thr).astype(int)
    out = df.copy()
    out["purchase_proba"] = proba
    out["purchase_pred"] = label
    return out

# ---------------------------
# Actions
# ---------------------------
a, b = st.columns(2)

with a:
    if st.button("🔮 Predict (single row)"):
        try:
            res = predict_df(single_input, threshold)
            verdict = "Will Purchase ✅" if int(res.loc[0, "purchase_pred"]) == 1 else "Unlikely to Purchase ❌"
            st.subheader("Prediction Result")
            st.success(f"{verdict} — Probability: **{res.loc[0, 'purchase_proba']:.3f}**")
        except HfHubHTTPError as e:
            st.error(f"Hub access error — if the model repo is private, add an HF_TOKEN secret. Details: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with b:
    uploaded = st.file_uploader("📦 Batch Scoring — upload CSV (same schema, no target)", type=["csv"])
    if uploaded and st.button("Run Batch Prediction"):
        try:
            df_in = pd.read_csv(io.BytesIO(uploaded.read()))
            res = predict_df(df_in, threshold)
            st.success("Batch predictions complete. Showing first 50 rows.")
            st.dataframe(res.head(50), use_container_width=True)
            st.download_button("⬇️ Download predictions", data=res.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.caption("If loading fails with 404, verify **MODEL_REPO** and **MODEL_FILENAME** in Space settings.")
