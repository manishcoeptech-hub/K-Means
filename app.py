import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("K-Means Clustering Lab – Step-by-Step with Elbow Method")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "phase" not in st.session_state:
    st.session_state.phase = 0

if "show_elbow" not in st.session_state:
    st.session_state.show_elbow = False

if "point_index" not in st.session_state:
    st.session_state.point_index = 0

# -------------------------------------------------
# DATASET UPLOAD
# -------------------------------------------------
st.header("1. Upload Dataset")

file = st.file_uploader("Upload CSV file (2 numeric columns)", type=["csv"])
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

    show_calc = st.checkbox("Show elbow method calculations")

    if st.button("Show / Hide Elbow Graph"):
        st.session_state.show_elbow = not st.session_state.show_elbow

    max_k = min(6, n)
    inertia_values = []

    elbow_explanation = ""

    for k in range(1, max_k + 1):
        centroids = X[:k]
        sq_distances = []
        rows = []

        for i, point in enumerate(X):
            dists = []
            for c in centroids:
                dx = point[0] - c[0]
                dy = point[1] - c[1]
                sq = dx**2 + dy**2
                dists.append(sq)

            min_sq = min(dists)
            sq_distances.append(min_sq)

            if show_calc:
                rows.append({
                    "Point": i + 1,
                    "Squared Distances to Centroids": dists,
                    "Minimum Squared Distance": min_sq
                })

        inertia = sum(sq_distances)
        inertia_values.append(inertia)

        if show_calc:
            st.subheader(f"K = {k}")
            st.dataframe(pd.DataFrame(rows), height=180)
            st.write(f"Inertia = {inertia}")

    elbow_df = pd.DataFrame({
        "K": range(1, max_k + 1),
        "Inertia": inertia_values
    })

    # Intelligent explanation
    diffs = np.diff(inertia_values)
    elbow_k = np.argmin(np.abs(diffs)) + 1

    st.subheader("Elbow Method Interpretation")
    st.write(
        f"""
        Inertia decreases rapidly until K = {elbow_k}, after which the reduction becomes slower.
        This indicates that adding more clusters beyond this point does not significantly improve
        compactness.

        Therefore, K = {elbow_k + 1} is a reasonable choice for this dataset.
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
        "Chosen number of clusters (based on elbow method)",
        min_value=2,
        max_value=max_k,
        value=min(max_k, elbow_k + 1)
    )

    if st.button("Next"):
        st.session_state.phase = 1
        st.session_state.point_index = 0

        # Intelligent centroid initialization (K-Means++)
        centroids = []
        centroids.append(X[np.random.randint(0, n)])

        while len(centroids) < k:
            distances = np.array([
                min(np.linalg.norm(x - c) for c in centroids)
                for x in X
            ])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)

        st.session_state.centroids = np.array(centroids)
        st.session_state.clusters = np.full(n, -1)

# -------------------------------------------------
# K-MEANS STEP-BY-STEP
# -------------------------------------------------
if st.session_state.phase == 1:

    centroids = st.session_state.centroids
    clusters = st.session_state.clusters
    i = st.session_state.point_index

    if i < n:
        with left:
            st.header("4. K-Means Algorithm (Point-wise)")

            show_point_calc = st.checkbox("Show distance calculations for this point")

            point = X[i]
            st.subheader(f"Processing Point {i + 1}")
            st.write(f"Coordinates: {point}")

            rows = []
            distances = []

            for idx, c in enumerate(centroids):
                dx = point[0] - c[0]
                dy = point[1] - c[1]
                sq = dx**2 + dy**2
                dist = np.sqrt(sq)

                distances.append(dist)

                if show_point_calc:
                    rows.append({
                        "Centroid": f"C{idx}",
                        "x - cx": dx,
                        "y - cy": dy,
                        "(x-cx)^2": dx**2,
                        "(y-cy)^2": dy**2,
                        "Sum": sq,
                        "Distance": dist
                    })

            if show_point_calc:
                st.dataframe(pd.DataFrame(rows), height=220)

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

            ax.scatter(
                centroids[:, 0], centroids[:, 1],
                c="black", marker="x", label="Centroids"
            )

            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.legend(fontsize=8)
            st.pyplot(fig)

    else:
        with left:
            st.success("All points processed. K-Means clustering completed.")
