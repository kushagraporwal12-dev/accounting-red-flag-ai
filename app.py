import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(
    page_title="Accounting Red Flag Analytics",
    layout="wide"
)

# ====================================================
# LOAD MODEL
# ====================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("random_forest_final.joblib")
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_artifacts()

# ====================================================
# SIDEBAR
# ====================================================
st.sidebar.title("Accounting Red Flag AI")
st.sidebar.caption("Early-warning analytics dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Input Format", "About"]
)

# ====================================================
# INPUT FORMAT PAGE
# ====================================================
if page == "Input Format":
    st.title("Expected CSV Structure")

    sample_df = pd.DataFrame({
        "Firm": ["Delhivery", "Blue Dart"],
        "Year": [2022, 2022],
        "Revenue_Growth": [18.5, 12.3],
        "Receivable_Turnover": [6.2, 8.1],
        "CFO_to_NI": [0.85, 1.12],
        "Accruals_Ratio": [0.18, 0.09],
        "Interest_Coverage": [3.2, 6.5]
    })

    st.dataframe(sample_df, use_container_width=True)
    st.info("Upload a CSV with the above columns. All ratios must be numeric.")

# ====================================================
# DASHBOARD
# ====================================================
if page == "Dashboard":

    st.title("Accounting Risk Dashboard")

    uploaded_file = st.file_uploader("Upload Financial Statement CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        X = df.drop(columns=["Firm", "Year"], errors="ignore")
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[:, 1]
        pred = model.predict(X_scaled)

        output = df.copy()
        output["Red_Flag_Probability"] = prob
        output["Red_Flag"] = pred

        total_obs = len(output)
        flagged_obs = int(output["Red_Flag"].sum())
        avg_prob = output["Red_Flag_Probability"].mean()

        # ====================================================
        # KPI CARDS
        # ====================================================
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Observations", total_obs)
        k2.metric("Red Flags Detected", flagged_obs)
        k3.metric("Average Risk Score", round(avg_prob, 2))

        # ====================================================
        # TABS
        # ====================================================
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Overview", "Distributions", "Key Drivers", "Flagged Firms"]
        )

        # ---------------- OVERVIEW ----------------
        with tab1:
            st.subheader("Project Overview")

            st.write(
                "This dashboard applies machine learning to financial-statement ratios "
                "to identify potential accounting red flags. The model evaluates earnings "
                "quality, cash flow sustainability, accrual intensity, and revenue realisation "
                "efficiency to assign a probability-based risk score to each firm-year observation."
            )

            fig = px.histogram(
                output,
                x="Red_Flag_Probability",
                nbins=20,
                color_discrete_sequence=["#00c6ff"]
            )
            fig.update_layout(
                height=300,
                xaxis_title="Red Flag Probability",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "This distribution provides a high-level view of accounting risk across the dataset."
            )

        # ---------------- DISTRIBUTIONS ----------------
        with tab2:
            st.subheader("Distribution of Financial Ratios")

            melted = X.melt(var_name="Ratio", value_name="Value")

            fig = px.box(
                melted,
                x="Ratio",
                y="Value",
                color="Ratio",
                height=350
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "The box plots highlight variability and outliers in financial ratios, "
                "which often serve as early indicators of accounting or operational stress."
            )

        # ---------------- KEY DRIVERS ----------------
        with tab3:
            st.subheader("Drivers of Red Flag Classification")

            imp_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance")

            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                height=350
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            dominant_driver = imp_df.iloc[-1]["Feature"]

            st.caption(
                f"The chart shows the relative contribution of each financial ratio. "
                f"Across the dataset, **{dominant_driver}** emerges as the most influential "
                "driver shaping red-flag classification."
            )

        # ---------------- FLAGGED FIRMS ----------------
        with tab4:
            st.subheader("Firm-Year Observations Flagged as High Risk")

            def explain(row):
                reasons = []
                if row.get("Revenue_Growth", 0) > 20 and row.get("Receivable_Turnover", 10) < 5:
                    reasons.append("aggressive revenue growth not supported by receivable collections")
                if row.get("CFO_to_NI", 1) < 0.4:
                    reasons.append("weak operating cash flow relative to reported profits")
                if row.get("Accruals_Ratio", 0) > 0.25:
                    reasons.append("high reliance on accrual-based accounting adjustments")
                if row.get("Interest_Coverage", 10) < 1.2:
                    reasons.append("limited ability to service interest obligations")
                return "; ".join(reasons) if reasons else "moderate risk indicators"

            output["Reason"] = output.apply(explain, axis=1)

            flagged_df = output[output["Red_Flag"] == 1]
            st.dataframe(flagged_df, use_container_width=True)

            # --------- ELABORATED DESCRIPTIVE CONCLUSION ---------
            if flagged_obs == 0:
                st.caption(
                    "No firm-year observations were flagged as high risk. "
                    "This suggests that, based on the analysed financial ratios, "
                    "earnings quality and cash flow alignment appear relatively stable."
                )
            else:
                st.markdown("### Analytical Interpretation of Flagged Firms")

                for _, row in flagged_df.iterrows():
                    st.write(
                        f"• **{row['Firm']} ({int(row['Year'])})** has been flagged with a "
                        f"red-flag probability of **{row['Red_Flag_Probability']:.2f}**. "
                        f"This elevated risk score is primarily driven by {row['Reason']}. "
                        "While this does not imply misreporting, the observed financial patterns "
                        "indicate areas that may warrant closer review by auditors, investors, "
                        "or management."
                    )

    else:
        st.info("Upload a dataset to activate the dashboard.")

# ====================================================
# ABOUT
# ====================================================
if page == "About":
    st.title("About This Dashboard")

    st.write("""
    This dashboard uses a Random Forest machine learning model trained on financial-statement
    ratios to identify potential accounting red flags.

    The system is designed as an **early-warning and prioritisation tool**, intended to support
    professional judgment rather than replace it.
    """)
