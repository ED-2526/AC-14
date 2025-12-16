import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# ================================
#   LAPLACIAN SCORE — FUNCIÓ BASE
# ================================
def laplacian_score(X, n_neighbors=5):
    n_samples, n_features = X.shape

    # 1. Matriu de veïnatge (W)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)

    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in indices[i]:
            W[i, j] = 1
            W[j, i] = 1

    # 2. Matriu de grau (D)
    D = np.diag(W.sum(axis=1))

    # 3. Laplacià (L = D - W)
    L = D - W

    # 4. Vector de pesos
    D1 = D @ np.ones((n_samples, 1))

    scores = np.zeros(n_features)

    for i in range(n_features):
        f = X[:, i].reshape(-1, 1)

        # Normalització segons paper original
        f_hat = f - (f.T @ D1) / D1.sum()

        numerador = (f_hat.T @ L @ f_hat)
        denominador = (f_hat.T @ D @ f_hat)

        scores[i] = numerador / denominador

    return scores.ravel()


# ================================
#     1) APLICACIÓ AL DATASET
# ================================
df = pd.read_csv("treballadors_definitiu.csv")

cols = [
    "Age",
    "Gender",
    "self_employed",
    "tech_company",
    "remote_work",
    "no_employees",
    "treatment",
    "family_history",
    "obs_consequence",
]

X_lap = df[cols].dropna()  


X_lap_scaled = MinMaxScaler().fit_transform(X_lap.values)

scores = laplacian_score(X_lap_scaled)

df_scores = (
    pd.DataFrame({"variable": cols, "laplacian_score": scores})
    .sort_values("laplacian_score", ascending=True)  # més baix = millor
    .reset_index(drop=True)
)

# ================================
#     2) GRÀFICA LAPLACIAN
# ================================
plt.figure(figsize=(10, 6))
plt.barh(df_scores["variable"], df_scores["laplacian_score"])
plt.xlabel("Laplacian Score (menor = millor)")
plt.ylabel("Variables")
plt.title("Rànquing de Variables segons Laplacian Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n===== RÀNQUING FINAL DE VARIABLES (Laplacian Score) =====")
print(df_scores)

# ================================
#     3) TRIEM LES TOP K VARIABLES
# ================================
TOP_K = 6  # canvia-ho fàcilment si vols 5, 6...
top_vars = df_scores["variable"].head(TOP_K).tolist()
print(f"\nTOP {TOP_K} variables seleccionades:", top_vars)

df_top = df[top_vars].dropna()
scaler = MinMaxScaler()
X = scaler.fit_transform(df_top.values)


# ================================
#     4) AVALUACIÓ PER DIFERENTS K
#        (Elbow + Silhouette + CH + DBI)
# ================================
K_RANGE = range(2, 11)
metrics = []

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels_k = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    sil = silhouette_score(X, labels_k)
    ch = calinski_harabasz_score(X, labels_k)
    db = davies_bouldin_score(X, labels_k)

    metrics.append(
        {
            "K": k,
            "inertia": inertia,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
        }
    )

df_metrics = pd.DataFrame(metrics)
print("\n===== MÈTRIQUES PER A CADA K =====")
print(df_metrics.round(3))

# --- Gràfica Elbow (inèrcia) ---
plt.figure(figsize=(10, 6))
plt.plot(df_metrics["K"], df_metrics["inertia"], marker="o")
plt.title("Mètode del colze — TOP variables seleccionades")
plt.xlabel("Nombre de clústers (K)")
plt.ylabel("Inèrcia (suma de distàncies internes)")
plt.xticks(K_RANGE)
plt.grid(True)
plt.show()

# --- Gràfica Silhouette ---
plt.figure(figsize=(10, 6))
plt.plot(df_metrics["K"], df_metrics["silhouette"], marker="o")
plt.title("Silhouette mitjà per cada K — TOP variables seleccionades")
plt.xlabel("Nombre de clústers (K)")
plt.ylabel("Silhouette mitjà")
plt.xticks(K_RANGE)
plt.grid(True)
plt.show()

# Pots mirar df_metrics i les gràfiques i triar el K que et sembli millor.
# Per exemple, si decideixes K = 8:
K_OPT = 6  # <-- ajusta aquest valor segons les mètriques


# ================================
#     5) CLUSTERING FINAL AMB K_OPT
# ================================
kmeans_final = KMeans(n_clusters=K_OPT, random_state=42, n_init="auto")
labels = kmeans_final.fit_predict(X)

df_top["cluster"] = labels

print("\n===== Distribució de treballadors per clúster =====")
print(df_top["cluster"].value_counts())

# ================================
#     6) t-SNE EN 2D (VISUALITZACIÓ)
# ================================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1500,
    random_state=42,
    init="random",
)

embedding = tsne.fit_transform(X)

df_top["TSNE1"] = embedding[:, 0]
df_top["TSNE2"] = embedding[:, 1]

plt.figure(figsize=(10, 7))
for c in range(K_OPT):
    subset = df_top[df_top["cluster"] == c]
    plt.scatter(subset["TSNE1"], subset["TSNE2"], label=f"Clúster {c}", alpha=0.7)

plt.title(f"Clustering final (K = {K_OPT}) visualitzat amb t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True)
plt.show()

# =======================================================
# 1) PERFIL MITJÀ COMPARAT PER CLÚSTER
# =======================================================
print("\n===== Perfil mitjà de cada clúster =====")
cluster_profiles = df_top.groupby("cluster")[top_vars].mean().round(3)
print(cluster_profiles)

# =======================================================
# 2) DISTÀNCIA DE CADA PUNT ALS CENTROIDS
# =======================================================
centroids = kmeans_final.cluster_centers_

# funció per calcular distàncies per variable
def contribucio_distances_per_var(X_escalat, centroids, top_vars):
    """
    Retorna per cada punt:
      - distància a cada centroid
      - contribució de cada variable a la distància al seu centroid assignat
    """
    distancias = np.linalg.norm(X_escalat[:, None, :] - centroids[None, :, :], axis=2)

    contrib_var = []
    for i, point in enumerate(X_escalat):
        cluster_assigned = df_top["cluster"].iloc[i]
        centroid_assigned = centroids[cluster_assigned]
        # contribució per variable de la distància
        contrib = (point - centroid_assigned) ** 2
        contrib_var.append(contrib)

    contrib_var = np.array(contrib_var)
    return distancias, contrib_var

distances, contrib_vars = contribucio_distances_per_var(X, centroids, top_vars)

# Afegir distàncies al dataframe
df_top["dist_to_centroid"] = distances[np.arange(len(distances)), df_top["cluster"]]

print("\n===== exemples de distàncies per observació =====")
print(df_top[["cluster", "dist_to_centroid"]].head())

# =======================================================
# 3) IMPORTÀNCIA RELATIVA DE LES VARIABLES PER CLÚSTER
# =======================================================
centroids_df = pd.DataFrame(centroids, columns=top_vars)

# variància entre centroids per cada variable
var_between_centroids = centroids_df.std(axis=0).sort_values(ascending=False)

print("\n===== Variables amb més variància entre centroids =====")
print(var_between_centroids)

# =======================================================
# 4) COMPARACIÓ DE CADA CLÚSTER RESPECTE LA MITJANA GLOBAL
# =======================================================
global_mean = df_top[top_vars].mean()
delta_per_cluster = (cluster_profiles - global_mean).abs().round(3)

print("\n===== Diferències de cada clúster respecte la mitjana global =====")
print(delta_per_cluster)

# =======================================================
# 5) FUNCIÓ PER ANALITZAR UN NOU REGISTRE
# =======================================================
def analitzar_registre(nou_registre_dict):
    """
    Rep un diccionari amb mateixes claus que 'top_vars'.
    Retorna:
      - cluster assignat
      - distància a cada centroid
      - contribució de variables que apropen / allunyen del centroid
      - perfil mitjà del cluster assignat
    """
    # 1) convertir a DataFrame
    df_new = pd.DataFrame([nou_registre_dict])
    df_new = df_new[top_vars]  # assegurar l'ordre de variables

    # 2) escalar
    X_new = scaler.transform(df_new.values)

    # 3) predir cluster
    cluster_pred = int(kmeans_final.predict(X_new)[0])

    # 4) distàncies a tots els centroids
    dists_to_centroids = np.linalg.norm(X_new - centroids, axis=1)

    # 5) contribució per variables al cluster assignat
    centroid_assigned = centroids[cluster_pred]
    contrib_new = (X_new[0] - centroid_assigned) ** 2
    contrib_series = pd.Series(contrib_new, index=top_vars).sort_values(ascending=False)

    # 6) perfil mitjà del cluster assignat
    profile_assigned = cluster_profiles.loc[cluster_pred]

    result = {
        "cluster_assigned": cluster_pred,
        "dists_to_centroids": dists_to_centroids,
        "variable_contributions": contrib_series,
        "profile_assigned": profile_assigned,
    }
    return result

# Exemple d'ús:
nou = {
    'family_history': 1,
    'Gender': 1,
    'self_employed': 0,
    'tech_company': 1,
    'remote_work': 1,
    'treatment': 1,
    'obs_consequence': 0,
}

resultat = analitzar_registre(nou)
print("Cluster assignat:", resultat["cluster_assigned"])
print("Distàncies a cada centroid:", resultat["dists_to_centroids"])
print("Contribució de variables (de més important a menys):")
print(resultat["variable_contributions"])
print("Perfil mitjà del cluster assignat:")
print(resultat["profile_assigned"])


global_mean = df_top[top_vars].mean()

for c in range(K_OPT):
    perfil = cluster_profiles.loc[c]
    diff = (perfil - global_mean).abs().sort_values(ascending=False)
    print(f"\nClúster {c} — variables més distintives:")
    print(diff)

