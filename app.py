
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛍",
    layout="wide"
)

# -------------------
# LOAD MODELS
# -------------------
kmeans = joblib.load("kmeans_model.joblib")
scaler = joblib.load("scaler.joblib")
similarity_df = joblib.load("recommendation_model.joblib")

# -------------------
# SIDEBAR
# -------------------
st.sidebar.title("🛍 Retail Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Product Recommendation"
    ]
)

# -------------------
# HOME
# -------------------
if page == "Home":

    st.title("🛍 Retail Analytics Dashboard")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Customers","4,372")
    col2.metric("Products","3,684")
    col3.metric("Transactions","22,190")
    col4.metric("Revenue","£8.9M")

    st.markdown("---")

    st.subheader("Project Objective")

    st.write('''
    This project uses:

    - K-Means Clustering for Customer Segmentation
    - Collaborative Filtering for Product Recommendation

    to improve customer retention and increase revenue.
    ''')

# -------------------
# CUSTOMER SEGMENTATION
# -------------------
elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    recency = st.number_input(
        "Recency",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency",
        min_value=1,
        value=10
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Segment"):

        log_freq = np.log1p(frequency)
        log_monetary = np.log1p(monetary)

        data = pd.DataFrame({
            "Recency":[recency],
            "LogFrequency":[log_freq],
            "LogMonetary":[log_monetary]
        })

        scaled = scaler.transform(data)

        cluster = kmeans.predict(scaled)[0]

        cluster_names = {
            0:"At Risk Customer",
            1:"Loyal Customer",
            2:"High Value Customer",
            3:"Low Value Customer"
        }

        st.success(
            f"Predicted Cluster: {cluster}"
        )

        st.info(
            cluster_names.get(
                cluster,
                "Customer Segment"
            )
        )

# -------------------
# PRODUCT RECOMMENDATION
# -------------------
elif page == "Product Recommendation":

    st.title("🎯 Product Recommendation")

    product = st.selectbox(
        "Select Product",
        similarity_df.columns
    )

    if st.button("Recommend"):

        recommendations = (
            similarity_df[product]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        st.subheader(
            "Recommended Products"
        )

        for item in recommendations.index:
            st.success(item)

