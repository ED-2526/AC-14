# ============================================
# CONFIGURACIÓ NO INTERACTIVA (IMPORTANT)
# ============================================
import matplotlib
matplotlib.use("Agg")  # evita bloquejos de plt.show()

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE

# ============================================
# 1. CARREGAR DADES
# ============================================
df = pd.read_csv("treballadors_definitiu.csv")

# Variables seleccionades via Laplacian Score (TOP 6)
vars_selected = [
    "family_history",
    "treatment",
    "remote_work",
    "Gender",
    "tech_company",
    "obs_consequence",
]

df_sel = df[vars_selected].dropna().copy()

# ============================================
# 2. ESCALAT
# ============================================
scaler = MinMaxScaler()
X = scaler.fit_transform(df_sel.values)

# ============================================
# 3. K-MEANS FINAL (K = 6)
# ============================================
K = 6
kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X)

df_sel["cluster"] = labels

# ============================================
# 4. DISTÀNCIA AL CENTROID
# ============================================
centroids = kmeans.cluster_centers_
distances = euclidean_distances(X, centroids)
df_sel["dist_to_centroid"] = distances[np.arange(len(distances)), labels]

# Percentil 75 per definir "perifèrics"
threshold = df_sel["dist_to_centroid"].quantile(0.75)
df_sel["posicio"] = np.where(
    df_sel["dist_to_centroid"] > threshold,
    "periferic",
    "central",
)

# ============================================
# 5. BOXPLOT DISTÀNCIA AL CENTROID (CLÚSTERS)
# ============================================
plt.figure(figsize=(10, 6))
sns.boxplot(x="cluster", y="dist_to_centroid", data=df_sel)
plt.title("Distribució de la distància al centroid per clúster")
plt.xlabel("Clúster")
plt.ylabel("Distància al centroid")
plt.tight_layout()
plt.savefig("01_distancia_centroid_clusters.png", dpi=200)
plt.close()

# ============================================
# 6. t-SNE (CENTRALS vs PERIFÈRICS)
# ============================================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1500,
    random_state=42,
)

embedding = tsne.fit_transform(X)
df_sel["TSNE1"] = embedding[:, 0]
df_sel["TSNE2"] = embedding[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_sel,
    x="TSNE1",
    y="TSNE2",
    hue="cluster",
    style="posicio",
    palette="tab10",
    alpha=0.8,
)
plt.title("t-SNE amb KMeans (centrals vs perifèrics)")
plt.tight_layout()
plt.savefig("02_tsne_centrals_periferics.png", dpi=200)
plt.close()

# ============================================
# 7. VARIABLES QUE EXPLIQUEN SER PERIFÈRIC
#    (comparació centrals vs perifèrics)
# ============================================
summary = []

for var in vars_selected:
    mean_central = df_sel[df_sel["posicio"] == "central"][var].mean()
    mean_periferic = df_sel[df_sel["posicio"] == "periferic"][var].mean()

    summary.append(
        {
            "variable": var,
            "mitjana_central": round(mean_central, 3),
            "mitjana_periferic": round(mean_periferic, 3),
            "diferencia": round(mean_periferic - mean_central, 3),
        }
    )

df_summary = pd.DataFrame(summary).sort_values(
    "diferencia", key=abs, ascending=False
)

df_summary.to_csv("03_variables_periferia.csv", index=False)

# ============================================
# 8. BOXPLOTS PER VARIABLE (CLÚSTER x POSICIÓ)
# ============================================
for var in vars_selected:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_sel,
        x="cluster",
        y=var,
        hue="posicio",
    )
    plt.title(f"{var} — diferències entre centrals i perifèrics")
    plt.tight_layout()
    plt.savefig(f"04_{var}_central_vs_periferic.png", dpi=200)
    plt.close()

# ============================================
# 9. RESUM FINAL PER PANTALLA
# ============================================
print("\n===== DISTRIBUCIÓ CENTRALS / PERIFÈRICS =====")
print(df_sel["posicio"].value_counts())

print("\n===== VARIABLES QUE MÉS EXPLIQUEN LA PERIFÈRIA =====")
print(df_summary)
