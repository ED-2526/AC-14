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

X_lap = df[cols].dropna()  # quasi no perds dades
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
TOP_K = 7  # canvia-ho fàcilment si vols 5, 6...
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
K_OPT = 8  # <-- ajusta aquest valor segons les mètriques


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

# ================================
#     7) PERFIL MITJÀ DE CADA CLÚSTER
# ================================
cluster_profiles = df_top.groupby("cluster")[top_vars].mean().round(3)

print(f"\n===== Perfil mitjà de cada clúster (TOP {TOP_K} variables) =====")
print(cluster_profiles)


# ================================
#     8) FUNCIÓ PER CLASSIFICAR NOUS INDIVIDUS
# ================================
def assignar_cluster(nou_registre: dict):
    """
    Rep un diccionari amb les mateixes claus que 'top_vars'
    (per ex. {"Age": 30, "Gender": 1, ...}) i retorna:

      - cluster assignat (int)
      - el perfil mitjà d'aquest clúster (Series)
    """

    # Convertim a DataFrame amb una sola fila
    df_new = pd.DataFrame([nou_registre])

    # Assegurem l'ordre de les columnes
    df_new = df_new[top_vars]

    # Apliquem el mateix escalat que a l'entrenament
    X_new = scaler.transform(df_new.values)

    # Prediem el clúster amb el KMeans entrenat
    cluster_pred = int(kmeans_final.predict(X_new)[0])

    # Recuperem el perfil mitjà d'aquest clúster
    perfil_cluster = cluster_profiles.loc[cluster_pred]

    return cluster_pred, perfil_cluster


# Exemple d'ús (comenta-ho o adapta-ho al teu cas):
"""
nou = {
    "Age": 30,
    "Gender": 1,
    "self_employed": 2,
    "tech_company": 1,
    "remote_work": 0,
    "no_employees": 3,
    "treatment": 0,
}
cluster_id, perfil = assignar_cluster(nou)
print("Nou individu assignat al clúster:", cluster_id)
print("Perfil mitjà del clúster:")
print(perfil)
"""
