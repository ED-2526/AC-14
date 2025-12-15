import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE

# ============================
# 1. Carregar el dataset
# ============================
df = pd.read_csv("treballadors_definitiu.csv")

# Variables seleccionades per tu (sense NaNs i sense soroll)
vars_selected = [
    "Age", 
    "Gender", 
    "self_employed", 
    "tech_company", 
    "remote_work", 
    "no_employees", 
    "treatment", 
    "family_history", 
    "obs_consequence"
]

# Filtrant les variables seleccionades
df_selected = df[vars_selected].dropna()  # eliminant les files amb NaNs

# ============================
# 2. Càlcul de Laplacian Score
# ============================
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

# Normalitzant les dades
X_scaled = MinMaxScaler().fit_transform(df_selected.values)

# Càlcul del Laplacian Score
scores = laplacian_score(X_scaled)

# Crear DataFrame per veure el rànquing
df_scores = (
    pd.DataFrame({"variable": vars_selected, "laplacian_score": scores})
    .sort_values("laplacian_score", ascending=True)  # més baix = millor
    .reset_index(drop=True)
)

# Mostrar el rànquing de variables
print("\n===== RÀNQUING FINAL DE VARIABLES (Laplacian Score) =====")
print(df_scores)

# ============================
# 3. Seleccionar TOP variables (les més importants)
# ============================
TOP_K = 6  # Pots canviar aquest valor si vols un nombre diferent de variables
top_vars = df_scores["variable"].head(TOP_K).tolist()
print(f"\nTOP {TOP_K} variables seleccionades:", top_vars)

# Filtrant les dades per les TOP variables
df_top = df[top_vars].dropna()

# Escalat de les dades per K-Means
scaler = MinMaxScaler()
X = scaler.fit_transform(df_top.values)

# ============================
# 4. Avaluació de K (Elbow, Silhouette, Calinski-Harabasz, Davies-Bouldin)
# ============================
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

# Gràfica Elbow (inèrcia)
plt.figure(figsize=(10, 6))
plt.plot(df_metrics["K"], df_metrics["inertia"], marker="o")
plt.title("Mètode del colze — TOP variables seleccionades")
plt.xlabel("Nombre de clústers (K)")
plt.ylabel("Inèrcia (suma de distàncies internes)")
plt.xticks(K_RANGE)
plt.grid(True)
plt.show()

# Gràfica Silhouette
plt.figure(figsize=(10, 6))
plt.plot(df_metrics["K"], df_metrics["silhouette"], marker="o")
plt.title("Silhouette mitjà per cada K — TOP variables seleccionades")
plt.xlabel("Nombre de clústers (K)")
plt.ylabel("Silhouette mitjà")
plt.xticks(K_RANGE)
plt.grid(True)
plt.show()

# Pot triar el millor K (per exemple, K=6)
K_OPT = 6

# ============================
# 5. K-Means final amb K_OPT
# ============================
kmeans_final = KMeans(n_clusters=K_OPT, random_state=42, n_init="auto")
labels = kmeans_final.fit_predict(X)

df_top["cluster"] = labels

print("\n===== Distribució de treballadors per clúster =====")
print(df_top["cluster"].value_counts())

# ============================
# 6. Visualització t-SNE en 2D
# ============================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1500,
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

# ============================
# 7. Perfils de cada clúster
# ============================
cluster_profiles = df_top.groupby("cluster")[top_vars].mean().round(3)

print(f"\n===== Perfil mitjà de cada clúster (TOP {TOP_K} variables) =====")
print(cluster_profiles)
