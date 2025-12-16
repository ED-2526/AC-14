import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.feature_selection import f_classif

import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("Set1")

# ============================================================
# 1. CARREGAR DATASET
# ============================================================
df_treballadors = pd.read_csv('treballadors_definitiu.csv')
print("Dataset carregat.")

# ============================================================
# 2. VARIABLES SELECCIONADES (6 MILLORS)
# ============================================================

variables_to_scale = []   # cap variable contínua
variables_no_scale = [
    'leave',
    'mental_health_consequence',
    'coworkers',
    'anonymity',
    'supervisor',
    'benefits'
]

variables_per_clustering = variables_no_scale.copy()

X_original = df_treballadors[variables_per_clustering].copy().dropna()
X = X_original.copy()

print("\nVariables utilitzades:", variables_per_clustering)
print("Registres utilitzats:", len(X))

# ============================================================
# 3. PREPARACIÓ DADES
# ============================================================

X_final = X.copy()
X_final_array = X_final.values

# ============================================================
# 4. CLUSTERING
# ============================================================

K = 4
models_info = []

kmeans = KMeans(n_clusters=K, random_state=42, n_init=15)
kmeans_labels = kmeans.fit_predict(X_final_array)
df_treballadors.loc[X_final.index, 'cluster_kmeans'] = kmeans_labels

# ============================================================
# 5. MÈTRIQUES
# ============================================================

def calcular_bss_tss(X_np, clusters):
    centroide_global = X_np.mean(axis=0)
    TSS = np.sum(np.sum((X_np - centroide_global)**2, axis=1))
    BSS = 0
    for c in np.unique(clusters):
        pts = X_np[clusters == c]
        centroide_c = pts.mean(axis=0)
        BSS += len(pts)*np.sum((centroide_c - centroide_global)**2)
    return BSS/TSS

dbs = davies_bouldin_score(X_final_array, kmeans_labels)
sil = silhouette_score(X_final_array, kmeans_labels)
ratio = calcular_bss_tss(X_final_array, kmeans_labels)

print("\nMÈTRIQUES:")
print("Davies-Bouldin:", dbs)
print("Silhouette:", sil)
print("BSS/TSS:", ratio)

# ============================================================
# 6. IMPORTÀNCIA DE VARIABLES
# ============================================================

f_vals, p_vals = f_classif(X_final, kmeans_labels)
importancia = pd.DataFrame({
    "variable": X_final.columns,
    "F_value": f_vals,
    "p_value": p_vals
}).sort_values("p_value")

print("\nIMPORTÀNCIA DE VARIABLES:")
print(importancia)

# ============================================================
# 7. RESUM PER CLÚSTER
# ============================================================

df_cl = X_original.copy()
df_cl["cluster"] = kmeans_labels

cluster_summary = df_cl.groupby("cluster").mean()
print("\nRESUM PER CLÚSTER:")
print(cluster_summary)

# ============================================================
# 8. HEATMAP
# ============================================================

plt.figure(figsize=(12,7))
norm = (cluster_summary - cluster_summary.mean()) / cluster_summary.std(ddof=0)
sns.heatmap(norm, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"Perfils psicològics - Clustering amb {len(variables_per_clustering)} variables")
plt.show()
