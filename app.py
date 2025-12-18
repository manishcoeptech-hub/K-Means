import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

st.title("Interactive K-Means Clustering Lab")
st.write("Students create a dataset, upload it, and explore K-Means step by step")

# -----------------------------
# Upload Dataset
# -----------------------------
st.header("1️⃣ Upload Your Dataset")

uploaded_file = st.file_uploader("Upload CSV file (2 numeric columns only)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df)

    if df.shape[1] != 2:
        st.error("Dataset must contain exactly 2 numeric columns.")
        st.stop()

    # -----------------------------
    # Choosing K
    # -----------------------------
    st.header("2️⃣ Choosing Number of Clusters (K)")

    max_k = min(10, len(df))
    inertia = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    fig_elbow, ax = plt.subplots()
    ax.plot(range(1, max_k + 1), inertia, marker='o')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Choosing K")

    st.pyplot(fig_elbow)

    st.markdown("""
    **Elbow Method Explanation:**
    - Inertia = Sum of squared distances of points from centroids
    - Choose K where inertia **starts decreasing slowly**
    """)

    # -----------------------------
    # Select K
    # -----------------------------
    k = st.slider("Select number of clusters (K)", 2, max_k, 3)

    # -----------------------------
    # Apply K-Means
    # -----------------------------
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df)
    centroids = kmeans.cluster_centers_

    # -----------------------------
    # Visualization
    # -----------------------------
    st.header("3️⃣ Cluster Visualization")

    fig, ax = plt.subplots()
    for cluster in range(k):
        cluster_data = df[df["Cluster"] == cluster]
        ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f"Cluster {cluster}")

    ax.scatter(centroids[:, 0], centroids[:, 1],
               c="black", s=200, marker="X", label="Centroids")

    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # Distance Calculation
    # -----------------------------
    st.header("4️⃣ Euclidean Distance Calculation")

    distances = pairwise_distances(df.iloc[:, :2], centroids, metric="euclidean")

    distance_df = pd.DataFrame(
        distances,
        columns=[f"Centroid {i}" for i in range(k)]
    )

    result_df = pd.concat([df.iloc[:, :2], distance_df, df["Cluster"]], axis=1)

    st.subheader("Distance of Each Point to Each Centroid")
    st.dataframe(result_df)

    # -----------------------------
    # Explanation
    # -----------------------------
    st.markdown("""
    **Euclidean Distance Formula:**

    \\[
    d = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
    \\]

    **Cluster Assignment Rule:**
    - Each point is assigned to the centroid with **minimum distance**
    """)

else:
    st.info("Please upload a CSV file to begin.")