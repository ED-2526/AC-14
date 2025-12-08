import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

plt.style.use('default')
sns.set_palette("husl")

# ==========================================================
# 1. LLEGIR I PREPARAR DATASET
# ==========================================================
df = pd.read_csv("treballadors_definitiu.csv")
print("Dataset carregat:", df.shape)

# Variables seleccionades per clustering
variables = [
    'Age', 'Gender',
    'treatment', 'work_interfere',
    'benefits', 'care_options', 'wellness_program',
    'seek_help', 'anonymity',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor',
    'mental_health_interview', 'phys_health_interview',
    'mental_vs_physical', 'obs_consequence'
]

df_cluster = df[variables].copy()

# ==========================================================
# 2. IMPUTACIÓ DE NULS (mitjana)
# ==========================================================
imputer = SimpleImputer(strategy="mean")
df_cluster = pd.DataFrame(imputer.fit_transform(df_cluster), columns=df_cluster.columns)

# ==========================================================
# 3. ESCALAT DE VARIABLES
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

print("Dades preparades per clustering:", X_scaled.shape)

# ==========================================================
# 4. K-MEANS (4 clusters)
# ==========================================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_scaled)

df["cluster_kmeans"] = clusters_kmeans
print("\nMostres per cluster (K-Means):")
print(df["cluster_kmeans"].value_counts())

# ==========================================================
# 5. Gaussian Mixture Model (4 clusters)
# ==========================================================
gmm = GaussianMixture(n_components=4, random_state=42)
clusters_gmm = gmm.fit_predict(X_scaled)

df["cluster_gmm"] = clusters_gmm
print("\nMostres per cluster (GMM):")
print(df["cluster_gmm"].value_counts())

# ==========================================================
# 6. Clustering Jeràrquic (Ward, 4 clusters)
# ==========================================================
hier = AgglomerativeClustering(n_clusters=4, linkage="ward")
clusters_hier = hier.fit_predict(X_scaled)

df["cluster_hier"] = clusters_hier
print("\nMostres per cluster (Jeràrquic):")
print(df["cluster_hier"].value_counts())

# ==========================================================
# 7. PCA PER VISUALITZAR
# ==========================================================
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)

df["PC1"] = pca_coords[:, 0]
df["PC2"] = pca_coords[:, 1]

# ==========================================================
# 8. PLOT PCA — K-MEANS
# ==========================================================
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=df["PC1"], 
    y=df["PC2"], 
    hue=df["cluster_kmeans"],
    palette="viridis",
    s=80,
    alpha=0.75
)
plt.title("Clustering Treballadors (K-Means, PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.show()

# ==========================================================
# 9. PLOT PCA — COLOR PER TREATMENT
# ==========================================================
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=df["PC1"], 
    y=df["PC2"], 
    hue=df["treatment"],
    palette="coolwarm",
    s=80,
    alpha=0.75
)
plt.title("Treballadors — Necessitat de Tractament Psicològic (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="treatment (0=No,1=Sí,2=NS/NC)")
plt.show()

# ==========================================================
# 10. MOSTRAR MITJANES PER CLÚSTER
# ==========================================================
print("\n==============================")
print("MITJANES PER CLÚSTER (K-Means)")
print("==============================")

cluster_means = df.groupby("cluster_kmeans")[variables].mean()
print(cluster_means)

# ==========================================================
# 11. GUARDAR RESULTATS
# ==========================================================
df.to_csv("treballadors_clusters.csv", index=False)
print("\n[OK] Fitxer 'treballadors_clusters.csv' generat correctament.")
