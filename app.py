import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("K-Means Clustering Lab – Step-by-Step Algorithm Trace")

# -------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------
if "phase" not in st.session_state:
    st.session_state.phase = 0

if "show_elbow" not in st.session_state:
    st.session_state.show_elbow = False

if "point_index" not in st.session_state:
    st.session_state.point_index = 0

if "centroids" not in st.session_state:
    st.session_state.centroids = None

if "clusters" not in st.session_state:
    st.session_state.clusters = None

if "history" not in st.session_state:
    st.session_state.history = []

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

    if st.button("Show / Hide Elbow Graph"):
        st.session_state.show_elbow = not st.session_state.show_elbow

    max_k = min(6, n)
    inertia_values = []

    for k_temp in range(1, max_k + 1):
        centroids_temp = X[:k_temp]
        squared_distances = []

        rows = []
        for i, point in enumerate(X):
            dists = []
            for c in centroids_temp:
                dx = point[0] - c[0]
                dy = point[1] - c[1]
                sq = dx**2 + dy**2
                dists.append(sq)

            min_sq = min(dists)
            squared_distances.append(min_sq)

            if show_elbow_calc:
                rows.append({
                    "Point": i + 1,
                    "Squared distances to centroids": dists,
                    "Minimum squared distance": min_sq
                })

        inertia = sum(squared_distances)
        inertia_values.append(inertia)

        if show_elbow_calc:
            st.subheader(f"K = {k_temp}")
            st.dataframe(pd.DataFrame(rows), height=180)
            st.write(f"Inertia = {inertia}")

    elbow_df = pd.DataFrame({
        "K": range(1, max_k + 1),
        "Inertia": inertia_values
    })

    diffs = np.diff(inertia_values)
    elbow_index = np.argmin(np.abs(diffs))
    suggested_k = elbow_index + 2

    st.subheader("Elbow Method Interpretation")
    st.write(
        f"""
        The inertia decreases rapidly up to K = {suggested_k - 1}.
        After this point, the decrease becomes slower.

        This indicates that K = {suggested_k} provides a good balance
        between compact clusters and model simplicity for this dataset.
        """
    )

with right:
    if st.session_state.show_elbow:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(elbow_df["K"], elbow_df["Inertia"], marker="o")
        ax.set_xlabel("Number of clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

# -------------------------------------------------
# SELECT K
# -------------------------------------------------
with left:
    st.header("3. Select K")

    k = st.number_input(
        "Chosen number of clusters",
        min_value=2,
        max_value=max_k,
        value=min(max_k, suggested_k)
    )

    if st.button("Start K-Means"):
        st.session_state.phase = 1
        st.session_state.point_index = 0
        st.session_state.history = []

        # Intelligent centroid initialization (K-Means++)
        centroids = [X[np.random.randint(0, n)]]
        while len(centroids) < k:
            distances = np.array([
                min(np.linalg.norm(x - c) for c in centroids)
                for x in X
            ])
            centroids.append(X[np.argmax(distances)])

        st.session_state.centroids = np.array(centroids, dtype=float)
        st.session_state.clusters = np.full(n, -1)

# -------------------------------------------------
# K-MEANS STEP-BY-STEP WITH NAVIGATION
# -------------------------------------------------
if st.session_state.phase == 1:

    centroids = st.session_state.centroids.copy()
    clusters = st.session_state.clusters.copy()
    idx = st.session_state.point_index

    if idx < len(st.session_state.history):
        state = st.session_state.history[idx]
        centroids = state["centroids"].copy()
        clusters = state["clusters"].copy()

    with left:
        st.header("4. K-Means Algorithm (Point-wise)")

        show_calc = st.checkbox("Show distance calculations for this point")

        if idx < n:
            point = X[idx]
            st.subheader(f"Processing Point {idx + 1}")
            st.write(f"Coordinates: {point}")

            rows = []
            distances = []

            for j, c in enumerate(centroids):
                dx = point[0] - c[0]
                dy = point[1] - c[1]
                sq = dx**2 + dy**2
                dist = np.sqrt(sq)

                distances.append(dist)

                if show_calc:
                    rows.append({
                        "Centroid": f"C{j}",
                        "x - cx": dx,
                        "y - cy": dy,
                        "(x-cx)^2": dx**2,
                        "(y-cy)^2": dy**2,
                        "Sum": sq,
                        "Distance": dist
                    })

            if show_calc:
                st.dataframe(pd.DataFrame(rows), height=220)

            assigned_cluster = int(np.argmin(distances))
            st.write(f"Assigned to Cluster {assigned_cluster}")

            if idx == len(st.session_state.history):
                st.session_state.history.append({
                    "centroids": centroids.copy(),
                    "clusters": clusters.copy()
                })

            clusters[idx] = assigned_cluster

            for j in range(k):
                pts = X[clusters == j]
                if len(pts) > 0:
                    centroids[j] = pts.mean(axis=0)

            st.write("Updated Centroids")
            st.dataframe(pd.DataFrame(centroids, columns=df.columns), height=120)

        else:
            st.subheader("K-Means Completed")
            st.write("All points have been assigned to clusters.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous") and idx > 0:
                st.session_state.point_index -= 1
        with col2:
            if st.button("Next") and idx <= n:
                st.session_state.point_index += 1

    # -------------------------------------------------
    # GRAPH (ALWAYS VISIBLE)
    # -------------------------------------------------
    with right:
        fig, ax = plt.subplots(figsize=(4, 4))

        for j in range(k):
            pts = X[clusters == j]
            ax.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {j}")

            if idx >= n and len(pts) > 0:
                center = centroids[j]
                radius = np.max(np.linalg.norm(pts - center, axis=1))
                circle = plt.Circle(center, radius, fill=False, linestyle="--")
                ax.add_patch(circle)

        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c="black", marker="x", label="Centroids"
        )

        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.legend(fontsize=8)
        st.pyplot(fig)

    st.session_state.centroids = centroids
    st.session_state.clusters = clusters
