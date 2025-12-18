import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("K-Means Clustering Lab – Step-by-Step Algorithm Trace")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

if "point_index" not in st.session_state:
    st.session_state.point_index = 0

# -------------------------------------------------
# DATASET UPLOAD
# -------------------------------------------------
st.header("1. Upload Dataset")

file = st.file_uploader("Upload CSV file (2 numeric columns only)", type=["csv"])

if not file:
    st.stop()

df = pd.read_csv(file)

if df.shape[1] != 2:
    st.error("Dataset must contain exactly two numeric columns.")
    st.stop()

X = df.values
n = len(X)

st.dataframe(df, height=200)

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
left, right = st.columns([6, 4])

# -------------------------------------------------
# ELBOW METHOD
# -------------------------------------------------
with left:
    st.header("2. Elbow Method")

    show_elbow_calc = st.checkbox("Show elbow method calculations")

    inertia_values = []
    elbow_tables = {}

    max_k = min(6, n)

    for k in range(1, max_k + 1):
        centroids = X[:k]
        squared_distances = []

        rows = []
        for i, point in enumerate(X):
            dists = []
            for c in centroids:
                dx = point[0] - c[0]
                dy = point[1] - c[1]
                sq = dx**2 + dy**2
                dists.append(sq)

            min_sq = min(dists)
            squared_distances.append(min_sq)

            rows.append({
                "Point": i + 1,
                "Squared Distances": dists,
                "Min Squared Distance": min_sq
            })

        inertia = sum(squared_distances)
        inertia_values.append(inertia)
        elbow_tables[k] = rows

        if show_elbow_calc:
            st.subheader(f"K = {k}")
            st.write("Inertia Calculation (Sum of Squared Distances)")
            calc_df = pd.DataFrame(rows)
            st.dataframe(calc_df, height=180)
            st.write(f"Inertia = {inertia}")

    elbow_df = pd.DataFrame({
        "K": range(1, max_k + 1),
        "Inertia": inertia_values
    })

with right:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(elbow_df["K"], elbow_df["Inertia"], marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

# -------------------------------------------------
# SELECT K
# -------------------------------------------------
with left:
    st.header("3. Select K and Start K-Means")
    k = st.number_input("Selected number of clusters (K)", min_value=2, max_value=max_k, value=2)

    if st.button("Next"):
        st.session_state.step = 1
        st.session_state.centroids = X[:k].astype(float)
        st.session_state.clusters = np.full(n, -1)
        st.session_state.point_index = 0

# -------------------------------------------------
# K-MEANS STEP-BY-STEP
# -------------------------------------------------
if st.session_state.step == 1:

    centroids = st.session_state.centroids
    clusters = st.session_state.clusters
    i = st.session_state.point_index

    if i < n:

        with left:
            st.header("4. K-Means Algorithm (Point-wise)")

            show_calc = st.checkbox("Show distance calculations")

            point = X[i]
            st.subheader(f"Processing Point {i + 1}")
            st.write(f"Point coordinates: {point}")

            distances = []
            calc_rows = []

            for idx, c in enumerate(centroids):
                dx = point[0] - c[0]
                dy = point[1] - c[1]
                sq = dx**2 + dy**2
                dist = np.sqrt(sq)

                distances.append(dist)

                if show_calc:
                    calc_rows.append({
                        "Centroid": f"C{idx}",
                        "x - cx": dx,
                        "y - cy": dy,
                        "(x-cx)^2": dx**2,
                        "(y-cy)^2": dy**2,
                        "Sum": sq,
                        "Distance": dist
                    })

            if show_calc:
                st.dataframe(pd.DataFrame(calc_rows), height=200)

            assigned_cluster = int(np.argmin(distances))
            st.write(f"Assigned to Cluster {assigned_cluster}")

            clusters[i] = assigned_cluster

            # Update centroids
            for j in range(k):
                pts = X[clusters == j]
                if len(pts) > 0:
                    centroids[j] = pts.mean(axis=0)

            st.write("Updated Centroids")
            st.dataframe(pd.DataFrame(centroids, columns=df.columns), height=120)

            if st.button("Next Point"):
                st.session_state.point_index += 1

        with right:
            fig, ax = plt.subplots(figsize=(4, 4))

            for j in range(k):
                pts = X[clusters == j]
                ax.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {j}")

            ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x")
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.legend(fontsize=8)
            st.pyplot(fig)

    else:
        with left:
            st.success("All points processed. K-Means clustering completed.")
