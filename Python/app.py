import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("churn_model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predict whether a customer is likely to churn.")

st.markdown("---")

# ================= Sidebar =================

st.sidebar.title("📊 Dashboard Info")
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model")
st.sidebar.write("Logistic Regression")
st.sidebar.subheader("🎯 Accuracy")
st.sidebar.success("78.7%")
st.sidebar.subheader("📁 Dataset")
st.sidebar.write("IBM Telco Customer Churn")
st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Developer")
st.sidebar.write("Yashwanth Naik")


# =====================================================
# HELPER: Encode one row into model-ready features (Telco model)
# =====================================================
def encode_customer(gender, senior, partner, dependents, tenure, phone,
                     multiple, internet, security, backup, protection,
                     support, tv, movies, contract, paperless, payment,
                     monthly, total):

    input_data = {col: 0 for col in feature_columns}

    input_data["SeniorCitizen"] = senior
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total

    def set_feature(name):
        if name in input_data:
            input_data[name] = 1

    if gender == "Male":
        set_feature("gender_Male")
    if partner == "Yes":
        set_feature("Partner_Yes")
    if dependents == "Yes":
        set_feature("Dependents_Yes")
    if phone == "Yes":
        set_feature("PhoneService_Yes")

    if multiple == "Yes":
        set_feature("MultipleLines_Yes")
    elif multiple == "No phone service":
        set_feature("MultipleLines_No phone service")

    if internet == "Fiber optic":
        set_feature("InternetService_Fiber optic")
    elif internet == "No":
        set_feature("InternetService_No")

    if security == "Yes":
        set_feature("OnlineSecurity_Yes")
    elif security == "No internet service":
        set_feature("OnlineSecurity_No internet service")

    if backup == "Yes":
        set_feature("OnlineBackup_Yes")
    elif backup == "No internet service":
        set_feature("OnlineBackup_No internet service")

    if protection == "Yes":
        set_feature("DeviceProtection_Yes")
    elif protection == "No internet service":
        set_feature("DeviceProtection_No internet service")

    if support == "Yes":
        set_feature("TechSupport_Yes")
    elif support == "No internet service":
        set_feature("TechSupport_No internet service")

    if tv == "Yes":
        set_feature("StreamingTV_Yes")
    elif tv == "No internet service":
        set_feature("StreamingTV_No internet service")

    if movies == "Yes":
        set_feature("StreamingMovies_Yes")
    elif movies == "No internet service":
        set_feature("StreamingMovies_No internet service")

    if contract == "One year":
        set_feature("Contract_One year")
    elif contract == "Two year":
        set_feature("Contract_Two year")

    if paperless == "Yes":
        set_feature("PaperlessBilling_Yes")

    if payment == "Credit card (automatic)":
        set_feature("PaymentMethod_Credit card (automatic)")
    elif payment == "Electronic check":
        set_feature("PaymentMethod_Electronic check")
    elif payment == "Mailed check":
        set_feature("PaymentMethod_Mailed check")

    return input_data


def risk_label(probability):
    if probability >= 0.70:
        return "🔴 High Risk"
    elif probability >= 0.40:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"


# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs([
    "🧍 Single Customer Prediction",
    "📁 Bulk Prediction (Telco Format)",
    "🤖 Auto-Train on Any Dataset"
])

# =====================================================
# TAB 1 — SINGLE CUSTOMER PREDICTION
# =====================================================
with tab1:

    st.header("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    st.markdown("---")

    predict = st.button("🔍 Predict Churn", use_container_width=True)

    if predict:

        input_data = encode_customer(
            gender, senior, partner, dependents, tenure, phone,
            multiple, internet, security, backup, protection,
            support, tv, movies, contract, paperless, payment,
            monthly, total
        )

        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 Model Accuracy", "78.7%")

        with col2:
            st.metric("🤖 Prediction", "CHURN" if prediction == 1 else "NO CHURN")

        with col3:
            st.metric("📊 Probability", f"{probability*100:.2f}%")

        st.divider()

        st.subheader("Prediction Result")

        st.metric("Churn Probability", f"{probability*100:.2f}%")

        if probability >= 0.70:
            bar_color = "#F93030"
        elif probability >= 0.40:
            bar_color = "#FFC107"
        else:
            bar_color = "#21C55D"

        label = risk_label(probability)

        st.markdown(f"""
        <div style="background-color:#333; border-radius:10px; height:22px; width:100%;">
            <div style="background-color:{bar_color}; width:{probability*100}%; height:100%; border-radius:10px; text-align:center; color:white; font-size:13px; line-height:22px;">
                {probability*100:.1f}% — {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if probability >= 0.70:
            st.error("🔴 HIGH RISK CUSTOMER")
            st.markdown("""
### 💡 Recommended Actions
- 📞 Contact customer within 24 hours
- 🎁 Offer loyalty discount
- 📃 Suggest yearly contract
- 🛠️ Provide free technical support
""")

        elif probability >= 0.40:
            st.warning("🟡 MEDIUM RISK CUSTOMER")
            st.markdown("""
### 💡 Recommended Actions
- 📧 Send promotional offers
- 💬 Monitor customer activity
- 🎯 Recommend value-added services
""")

        else:
            st.success("🟢 LOW RISK CUSTOMER")
            st.markdown("""
### 💡 Recommended Actions
- ❤️ Reward customer loyalty
- 🎉 Offer premium membership
- 📩 Send appreciation email
""")

        report = pd.DataFrame({
            "Prediction": ["CHURN" if prediction == 1 else "NO CHURN"],
            "Probability (%)": [round(probability * 100, 2)],
            "Gender": [gender],
            "Senior Citizen": [senior],
            "Partner": [partner],
            "Dependents": [dependents],
            "Tenure": [tenure],
            "Monthly Charges": [monthly],
            "Total Charges": [total],
            "Contract": [contract]
        })

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="prediction_report.csv",
            mime="text/csv"
        )

        st.divider()


# =====================================================
# TAB 2 — BULK CSV UPLOAD (FIXED TELCO MODEL)
# =====================================================
with tab2:

    st.header("Upload Customer Data (CSV)")
    st.write("Upload a CSV file with Telco-format columns to predict churn for all customers using the trained model.")

    REQUIRED_COLUMNS = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    with st.expander("📋 Required CSV columns"):
        st.code(", ".join(REQUIRED_COLUMNS))

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="bulk_telco")

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            data = None

        if data is not None:

            missing_cols = [c for c in REQUIRED_COLUMNS if c not in data.columns]

            if missing_cols:
                st.error("❌ This file doesn't match the expected schema.")
                st.write("**Missing columns:**")
                st.code(", ".join(missing_cols))
                st.info("This model was trained on the IBM Telco Customer Churn dataset format. "
                        "Try the 'Auto-Train on Any Dataset' tab instead for other formats.")

            else:
                st.success(f"✅ File looks good! Found {len(data)} customer records.")

                if st.button("🔍 Run Bulk Prediction", use_container_width=True):

                    with st.spinner("Running predictions..."):

                        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)

                        results = []

                        for _, row in data.iterrows():
                            input_data = encode_customer(
                                row["gender"], row["SeniorCitizen"], row["Partner"],
                                row["Dependents"], row["tenure"], row["PhoneService"],
                                row["MultipleLines"], row["InternetService"], row["OnlineSecurity"],
                                row["OnlineBackup"], row["DeviceProtection"], row["TechSupport"],
                                row["StreamingTV"], row["StreamingMovies"], row["Contract"],
                                row["PaperlessBilling"], row["PaymentMethod"],
                                row["MonthlyCharges"], row["TotalCharges"]
                            )

                            input_df = pd.DataFrame([input_data])[feature_columns]

                            pred = model.predict(input_df)[0]
                            prob = model.predict_proba(input_df)[0][1]

                            results.append({
                                "Prediction": "CHURN" if pred == 1 else "NO CHURN",
                                "Probability (%)": round(prob * 100, 2),
                                "Risk Level": risk_label(prob)
                            })

                        results_df = pd.DataFrame(results)
                        final_df = pd.concat([data.reset_index(drop=True), results_df], axis=1)

                    st.success("🎉 Predictions complete!")

                    total_customers = len(final_df)
                    churn_count = (final_df["Prediction"] == "CHURN").sum()
                    churn_rate = (churn_count / total_customers) * 100

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("👥 Total Customers", total_customers)
                    with m2:
                        st.metric("⚠️ Predicted Churns", churn_count)
                    with m3:
                        st.metric("📊 Churn Rate", f"{churn_rate:.1f}%")

                    st.divider()

                    st.subheader("Results")
                    st.dataframe(final_df, use_container_width=True)

                    csv_out = final_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="📥 Download Full Results CSV",
                        data=csv_out,
                        file_name="bulk_churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# =====================================================
# TAB 3 — AUTO-TRAIN ON ANY DATASET
# =====================================================
with tab3:

    st.header("🤖 Auto-Train a Model on Any Dataset")
    st.write(
        "Upload **any** customer dataset that already contains a churn/target column. "
        "This tool will automatically clean the data, train a brand-new model on it, "
        "and predict churn for every row."
    )

    st.info(
        "⚠️ This only works if your file already has a column showing past churn outcomes "
        "(e.g. a 'Churn' column with Yes/No or 1/0). The app cannot guess churn for data "
        "that has no historical outcome to learn from."
    )

    auto_file = st.file_uploader("Choose a CSV file", type=["csv"], key="auto_train")

    if auto_file is not None:

        try:
            auto_data = pd.read_csv(auto_file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            auto_data = None

        if auto_data is not None:

            st.subheader("📋 Data Preview")
            st.dataframe(auto_data.head(), use_container_width=True)
            st.caption(f"{auto_data.shape[0]} rows × {auto_data.shape[1]} columns")

            target_col = st.selectbox(
                "Which column represents churn (the target you want to predict)?",
                auto_data.columns
            )

            model_choice = st.radio(
                "Choose a model type",
                ["Logistic Regression", "Random Forest"],
                horizontal=True
            )

            train_btn = st.button("🚀 Train Model", use_container_width=True)

            if train_btn:

                with st.spinner("Cleaning data and training model..."):

                    df = auto_data.copy()

                    # Drop rows where target is missing
                    df = df.dropna(subset=[target_col])

                    # Encode target into 0/1
                    if df[target_col].dtype == "object":
                        unique_vals = sorted(df[target_col].unique())
                        if len(unique_vals) != 2:
                            st.error(
                                f"❌ The target column '{target_col}' has {len(unique_vals)} unique "
                                "values. This tool only supports binary (2-outcome) targets, like Yes/No."
                            )
                            st.stop()
                        target_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                        y = df[target_col].map(target_map)
                        st.caption(f"Target mapping used: {unique_vals[0]} → 0, {unique_vals[1]} → 1")
                    else:
                        unique_vals = sorted(df[target_col].dropna().unique())
                        if len(unique_vals) != 2:
                            st.error(
                                f"❌ The target column '{target_col}' has {len(unique_vals)} unique "
                                "values. This tool only supports binary (2-outcome) targets."
                            )
                            st.stop()
                        y = df[target_col]

                    X = df.drop(columns=[target_col])

                    # Drop ID-like columns (every value unique = useless for ML)
                    id_like_cols = [c for c in X.columns if X[c].nunique() == len(X)]
                    if id_like_cols:
                        X = X.drop(columns=id_like_cols)

                    # Drop constant columns (no useful info)
                    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
                    if constant_cols:
                        X = X.drop(columns=constant_cols)

                    # Try converting object columns that are secretly numeric
                    for c in X.columns:
                        if X[c].dtype == "object":
                            converted = pd.to_numeric(X[c], errors="coerce")
                            if converted.notna().sum() / len(converted) > 0.9:
                                X[c] = converted

                    # Fill missing values
                    for c in X.columns:
                        if X[c].dtype in [np.float64, np.int64]:
                            X[c] = X[c].fillna(X[c].median())
                        else:
                            X[c] = X[c].fillna("Unknown")

                    # One-hot encode remaining categorical columns
                    X_encoded = pd.get_dummies(X, drop_first=True)

                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=0.2, random_state=42, stratify=y
                    )

                    if model_choice == "Logistic Regression":
                        clf = LogisticRegression(max_iter=2000)
                    else:
                        clf = RandomForestClassifier(n_estimators=150, random_state=42)

                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    # Predict on full dataset for the results table
                    full_pred = clf.predict(X_encoded)
                    full_prob = clf.predict_proba(X_encoded)[:, 1]

                st.success("🎉 Model trained successfully!")

                if id_like_cols:
                    st.caption(f"Dropped ID-like columns automatically: {', '.join(id_like_cols)}")
                if constant_cols:
                    st.caption(f"Dropped constant columns automatically: {', '.join(constant_cols)}")

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("🎯 Test Accuracy", f"{acc*100:.2f}%")
                with m2:
                    st.metric("📊 Rows Used", f"{len(X_encoded)}")

                st.subheader("Confusion Matrix (on held-out test data)")
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual: No Churn", "Actual: Churn"],
                    columns=["Predicted: No Churn", "Predicted: Churn"]
                )
                st.dataframe(cm_df, use_container_width=True)

                st.divider()

                st.subheader("Predictions for All Customers")

                results_df = auto_data.loc[df.index].copy()
                results_df["Predicted Churn"] = ["CHURN" if p == 1 else "NO CHURN" for p in full_pred]
                results_df["Churn Probability (%)"] = (full_prob * 100).round(2)
                results_df["Risk Level"] = [risk_label(p) for p in full_prob]

                churn_count = (results_df["Predicted Churn"] == "CHURN").sum()
                churn_rate = (churn_count / len(results_df)) * 100

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("👥 Total Customers", len(results_df))
                with m2:
                    st.metric("⚠️ Predicted Churns", churn_count)
                with m3:
                    st.metric("📊 Churn Rate", f"{churn_rate:.1f}%")

                st.dataframe(results_df, use_container_width=True)

                csv_out = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_out,
                    file_name="auto_trained_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                model_bytes = pickle.dumps(clf)
                st.download_button(
                    label="📦 Download Trained Model (.pkl)",
                    data=model_bytes,
                    file_name="auto_trained_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )


# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:15px;'>

Developed by <b>Yashwanth Naik</b><br>


Python • Streamlit • Scikit-Learn • Machine Learning

</div>
""", unsafe_allow_html=True)