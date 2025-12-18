import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

st.set_page_config(layout="wide")
st.title("K-Means Clustering – Complete Step-by-Step Visualization")

# -------------------------------------------------
# Upload Dataset
# -------------------------------------------------
st.header("1️⃣ Upload Your Dataset")

file = st.file_uploader("Upload CSV file (2 numeric columns)", type=["csv"])

if file:
    df = pd.read_csv(file)

    if df.shape[1] != 2:
        st.error("Dataset must have exactly 2 numeric columns.")
        st.stop()

    X = df.values
    st.dataframe(df)

    # -------------------------------------------------
    # ELBOW METHOD (FULL CALCULATIONS)
    # -------------------------------------------------
    st.header("2️⃣ Elbow Method (With Full Calculations)")

    max_k = min(6, len(X))
    inertia = {}

    for k in range(1, max_k + 1):
        centroids = X[:k]
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        min_dist = np.min(distances, axis=1)
        inertia[k] = np.sum(min_dist ** 2)

    if st.button("Show Elbow Method Calculations"):
        elbow_df = pd.DataFrame({
            "K": list(inertia.keys()),
            "Inertia (Sum of Squared Distances)": list(inertia.values())
        })
        st.dataframe(elbow_df)

        fig, ax = plt.subplots()
        ax.plot(elbow_df["K"], elbow_df.iloc[:, 1], marker="o")
        ax.set_xlabel("K")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        st.markdown("""
        **Inertia Formula:**

        \\[
        \\text{Inertia} = \\sum_{i=1}^{n} (d_i)^2
        \\]

        where \( d_i \) is the minimum Euclidean distance
        from point \( i \) to its nearest centroid.
        """)

    # -------------------------------------------------
    # SELECT K
    # -------------------------------------------------
    st.header("3️⃣ Select Number of Clusters")
    k = st.slider("Choose K", 2, max_k, 3)

    # -------------------------------------------------
    # MANUAL K-MEANS ITERATIONS
    # -------------------------------------------------
    st.header("4️⃣ K-Means Iterations (All Calculations)")

    centroids = X[:k].copy()

    for iteration in range(1, 4):
        st.subheader(f"Iteration {iteration}")

        # Distance calculation
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)

        distance_df = pd.DataFrame(
            distances,
            columns=[f"Centroid {i}" for i in range(k)]
        )
        st.write("📐 Euclidean Distance of Each Point to Each Centroid")
        st.dataframe(distance_df)

        # Cluster assignment
        clusters = np.argmin(distances, axis=1)
        assign_df = df.copy()
        assign_df["Cluster"] = clusters
        st.write("📌 Cluster Assignment")
        st.dataframe(assign_df)

        # Update centroids
        new_centroids = np.array([
            X[clusters == i].mean(axis=0) for i in range(k)
        ])

        centroid_df = pd.DataFrame(
            new_centroids,
            columns=df.columns,
            index=[f"Centroid {i}" for i in range(k)]
        )

        st.write("🔄 Updated Centroids")
        st.dataframe(centroid_df)

        if np.allclose(centroids, new_centroids):
            st.success("Centroids converged. Algorithm stops.")
            break

        centroids = new_centroids

    # -------------------------------------------------
    # FINAL CLUSTER VISUALIZATION
    # -------------------------------------------------
    st.header("5️⃣ Final Cluster Visualization")

    fig, ax = plt.subplots()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i in range(k):
        cluster_points = X[clusters == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   color=colors[i], label=f"Cluster {i}")

        # Draw circle around cluster
        center = centroids[i]
        radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
        circle = Circle(center, radius, fill=False, linestyle="--")
        ax.add_patch(circle)

    ax.scatter(centroids[:, 0], centroids[:, 1],
               marker="X", s=200, color="black", label="Centroids")

    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_title("K-Means Clusters with Boundary Circles")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Upload a dataset to start the lab.")
