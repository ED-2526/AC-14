import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# ============================================
# 1. CARREGAR DADES
# ============================================
df = pd.read_csv("treballadors_definitiu.csv")

# Variables TOP segons Laplacian Score
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
# 3. K-MEANS (K = 6)
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

# Definim perifèrics dins de cada clúster (percentil 75)
df_sel["posicio"] = "central"

for c in range(K):
    mask = df_sel["cluster"] == c
    threshold = df_sel.loc[mask, "dist_to_centroid"].quantile(0.75)
    df_sel.loc[mask & (df_sel["dist_to_centroid"] > threshold), "posicio"] = "periferic"

# ============================================
# 5. ANÀLISI PER CLÚSTER (NOMÉS DIFERÈNCIES POSITIVES)
# ============================================
results = []

for c in range(K):
    df_c = df_sel[df_sel["cluster"] == c]

    centrals = df_c[df_c["posicio"] == "central"]
    periferics = df_c[df_c["posicio"] == "periferic"]

    # Evitem clústers massa petits
    if len(centrals) < 5 or len(periferics) < 5:
        continue

    for var in vars_selected:
        mean_c = centrals[var].mean()
        mean_p = periferics[var].mean()
        diff = mean_p - mean_c

        # NOMÉS diferències positives
        if diff > 0:
            results.append({
                "cluster": c,
                "variable": var,
                "mitjana_central": round(mean_c, 3),
                "mitjana_periferic": round(mean_p, 3),
                "diferencia": round(diff, 3),
            })

# ============================================
# 6. TAULA FINAL: TOP 2 VARIABLES PER CLÚSTER
# ============================================
df_results = pd.DataFrame(results)

df_final = (
    df_results
    .sort_values(["cluster", "diferencia"], ascending=[True, False])
    .groupby("cluster")
    .head(2)
    .reset_index(drop=True)
)

print("\n===== VARIABLES CONTEXTUALS QUE EXPLIQUEN LA PERIFÈRIA (PER CLÚSTER) =====\n")
print(df_final)
