import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ================================
# 1. Carregar dataset net
# ================================
df = pd.read_csv("treballadors_definitiu.csv")

variables_importants = [
    'Age',
    'no_employees',
    'work_interfere',
    'leave',
    'care_options',
    'benefits',
    'family_history',
    'mental_health_consequence',
    'supervisor',
    'coworkers',
    'phys_health_interview',
    'mental_vs_physical',
    'seek_help',
    'anonymity',
    'wellness_program'
]

df_clean = df[variables_importants].dropna()

# ================================
# 2. Escalar dades
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# ================================
# 3. PCA (per inspecció)
# ================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variància explicada per PCA:", pca.explained_variance_ratio_)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], s=10)
plt.title("PCA - Distribució de les dades")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# ================================
# 4. Elbow Method
# ================================
distortions = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(K, distortions, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Distortion (inertia)")
plt.show()

# ================================
# 5. Silhouette per cada k
# ================================
silhouette_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(7,5))
plt.plot(K, silhouette_scores, marker='o')
plt.title("Silhouette Score per k")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.show()

print("Silhouette scores:", silhouette_scores)

# ================================
# 6. Triem valor òptim de k
# ================================
k_opt = K[silhouette_scores.index(max(silhouette_scores))]
print("k òptim segons silhouette:", k_opt)

# ================================
# 7. Clustering final
# ================================
kmeans_final = KMeans(n_clusters=k_opt, random_state=0)
labels_final = kmeans_final.fit_predict(X_scaled)

df_clean["cluster"] = labels_final

print(df_clean.head())

# ================================
# 8. Visualització final amb PCA
# ================================
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_final, cmap='viridis', s=10)
plt.title("Clusters en PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
