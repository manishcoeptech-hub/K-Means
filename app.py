import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("K-Means Clustering: Step-by-Step Algorithm Trace")

# ----------------------------
# Upload Dataset
# ----------------------------
file = st.file_uploader("Upload CSV file (2 numeric columns)", type=["csv"])

if not file:
    st.stop()

df = pd.read_csv(file)

if df.shape[1] != 2:
    st.error("Dataset must contain exactly 2 numeric columns.")
    st.stop()

X = df.values
st.dataframe(df, height=200)

# ----------------------------
# Select K
# ----------------------------
k = st.number_input("Number of clusters (K)", min_value=2, max_value=5, value=2)

# ----------------------------
# Toggle Calculations
# ----------------------------
show_calc = st.checkbox("Show distance calculations", value=False)

# ----------------------------
# Initialize Centroids
# ----------------------------
centroids = X[:k].astype(float)
clusters = np.full(len(X), -1)

st.subheader("Initial Centroids")
st.dataframe(
    pd.DataFrame(centroids, columns=df.columns),
    height=120
)

# ----------------------------
# Algorithm Trace
# ----------------------------
st.subheader("Point-wise Processing")

for i, point in enumerate(X):

    st.write(f"Processing point {i + 1}: {point}")

    distances = []
    for c in centroids:
        d = np.sqrt(np.sum((point - c) ** 2))
        distances.append(d)

    if show_calc:
        calc_df = pd.DataFrame({
            "Centroid": [f"C{j}" for j in range(k)],
            "Distance": distances
        })
        st.dataframe(calc_df, height=120)

    new_cluster = int(np.argmin(distances))
    old_cluster = clusters[i]
    clusters[i] = new_cluster

    st.write(f"Assigned to Cluster {new_cluster}")

    # Update centroid incrementally
    if old_cluster != new_cluster:
        for j in range(k):
            pts = X[clusters == j]
            if len(pts) > 0:
                centroids[j] = pts.mean(axis=0)

        if show_calc:
            st.write("Updated Centroids")
            st.dataframe(
                pd.DataFrame(centroids, columns=df.columns),
                height=120
            )

    st.markdown("---")

# ----------------------------
# Final Cluster Plot (Small)
# ----------------------------
st.subheader("Final Clusters")

fig, ax = plt.subplots(figsize=(4, 4))
for j in range(k):
    pts = X[clusters == j]
    ax.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {j}")

ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x")
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])
ax.legend(fontsize=8)
st.pyplot(fig)
