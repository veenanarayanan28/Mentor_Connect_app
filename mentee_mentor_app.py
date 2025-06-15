# mentee_mentor_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("Mentee-Mentor Matching App")

# Load Data
mentor_df = pd.read_csv('mentors_database2.csv')
mentee_df = pd.read_csv('mentee_database.csv', delimiter=';')

# Preprocessing
mentor_df['Mentor_BurnoutRiskScore'] = mentor_df['Mentor_BurnoutRiskScore'].str.extract(r'(\d+)').astype(float)
mentee_df['Mentee_BurnoutRiskScore'] = mentee_df['Mentee_BurnoutRiskScore'].str.extract(r'(\d+)').astype(float)

# One-hot encode
categorical_mentors = ["Mentor_InterestField", "Mentor_Location"]
mentor_encoder = OneHotEncoder(sparse_output=False)
mentors_encoded = mentor_encoder.fit_transform(mentor_df[categorical_mentors])
numerical_mentors_scaled = StandardScaler().fit_transform(mentor_df[['Mentor_BurnoutRiskScore']])
X_mentors = np.hstack([numerical_mentors_scaled, mentors_encoded])

categorical_mentees = ["Mentee_InterestField", "Mentee_Location"]
mentee_encoder = OneHotEncoder(sparse_output=False)
mentees_encoded = mentee_encoder.fit_transform(mentee_df[categorical_mentees])
numerical_mentees_scaled = StandardScaler().fit_transform(mentee_df[['Mentee_BurnoutRiskScore']])
X_mentees = np.hstack([numerical_mentees_scaled, mentees_encoded])

# Clustering
sil_scores = []
k_values = range(2, 11)  # limit for speed

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_mentees)
    score = silhouette_score(X_mentees, labels)
    sil_scores.append(score)

best_k = k_values[np.argmax(sil_scores)]
st.write(f"Best number of mentee clusters: {best_k}")

# Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=42)
mentee_df['Cluster'] = kmeans.fit_predict(X_mentees)

# Similarity
cluster_centroids = []
for cluster_id in range(best_k):
    cluster_points = X_mentees[mentee_df['Cluster'] == cluster_id]
    centroid = cluster_points.mean(axis=0)
    cluster_centroids.append(centroid)
cluster_centroids = np.array(cluster_centroids)
similarity_matrix = cosine_similarity(cluster_centroids, X_mentors)
top_5_mentors_per_cluster = np.argsort(similarity_matrix, axis=1)[:, -5:][:, ::-1]

# Display Results
for cluster_id in range(best_k):
    st.subheader(f"Top 5 mentors for Mentee Cluster {cluster_id}:")
    mentor_indices = top_5_mentors_per_cluster[cluster_id]
    mentor_ids = mentor_df.iloc[mentor_indices]['Mentor_ID'].values
    mentor_similarities = similarity_matrix[cluster_id, mentor_indices]
    for mentor_id, sim_score in zip(mentor_ids, mentor_similarities):
        st.write(f"Mentor_ID: {mentor_id}, Similarity: {sim_score:.4f}")

# Optional: plot heatmap
if st.checkbox("Show heatmap"):
    heatmap_data = pd.DataFrame()
    for cluster_id in range(best_k):
        mentor_indices = top_5_mentors_per_cluster[cluster_id]
        mentor_ids = mentor_df.iloc[mentor_indices]['Mentor_ID'].astype(str).values
        similarities = similarity_matrix[cluster_id, mentor_indices]
        temp_series = pd.Series(similarities, index=mentor_ids, name=f'Cluster {cluster_id}')
        heatmap_data = pd.concat([heatmap_data, temp_series.to_frame().T], axis=0)
    heatmap_data = heatmap_data.fillna(0)
    st.write("### Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', ax=ax)
    st.pyplot(fig)
