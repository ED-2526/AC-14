import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import seaborn as sns
import matplotlib.pyplot as plt

# ===========================
# 1. Carregar dataset
# ===========================
df = pd.read_csv("treballadors_definitiu.csv")

# ===========================
# 2. Variables que utilitzarem
# ===========================
vars_clustering = [
    "work_interfere",
    "family_history",
    "no_employees",
    "care_options",
    "benefits",
    "supervisor",
    "treatment",
    "remote_work",
    "seek_help",
    "wellness_program",
    "Gender",
    "phys_health_consequence",
    "mental_health_consequence",
    "tech_company"
]

df_cluster = df[vars_clustering].copy()

# ===========================
# 3. Tractar valors NaN
# ===========================
df_cluster = df_cluster.fillna("Unknown")

# Convertir tot a string (K-Modes ho requereix)
df_cluster = df_cluster.astype(str)

print("Dataframe final per clustering:", df_cluster.shape)

# ===========================
# 4. Aplicar K-Modes
# ===========================
k = 5  # pots variar-ho després
km = KModes(n_clusters=k, init="Huang", n_init=10, verbose=1)
clusters = km.fit_predict(df_cluster)

df_cluster["cluster"] = clusters

print("\nMida de cada cluster:")
print(df_cluster["cluster"].value_counts())

# ===========================
# 5. Calcular perfil (mode per cluster)
# ===========================
perfil_clusters = df_cluster.groupby("cluster").agg(lambda x: x.mode()[0])
print("\nPerfils de clusters:")
print(perfil_clusters)

# ===========================
# 6. Heatmap dels perfils
# ===========================
plt.figure(figsize=(14,8))
sns.heatmap(
    perfil_clusters.replace("Unknown", np.nan).astype(float),
    cmap="Spectral",
    annot=True,
    fmt=".2f"
)
plt.title("Heatmap de perfils dels clusters (K-Modes)")
plt.show()
