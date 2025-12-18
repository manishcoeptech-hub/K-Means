import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("K-Means Clustering Lab")
st.write("Illustration of K-Means Clustering on a Small Dataset")

# Sidebar
st.sidebar.header("Controls")
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=5, value=3)

# Create small dataset
data = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8],
    [1, 0.6], [9, 11], [8, 2], [10, 2],
    [9, 3], [4, 7]
])

df = pd.DataFrame(data, columns=["X", "Y"])

st.subheader("Dataset")
st.dataframe(df)

# Apply K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

# Plot
fig, ax = plt.subplots()
for cluster in range(k):
    cluster_data = df[df["Cluster"] == cluster]
    ax.scatter(cluster_data["X"], cluster_data["Y"], label=f"Cluster {cluster}")

# Plot centroids
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1],
           s=200, c='black', marker='X', label="Centroids")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("K-Means Clustering Visualization")
ax.legend()

st.pyplot(fig)

# Explanation
st.subheader("Explanation")
st.write("""
- K-Means divides data into **K clusters**
- Each cluster has a **centroid**
- Points are assigned to the nearest centroid
- Centroids move iteratively to minimize distance
""")