import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

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
#     APLICACIÓ AL TEU DATASET
# ================================
df = pd.read_csv("treballadors_definitiu.csv")

"""
cols = ["Age","Gender","self_employed","family_history",
        "treatment","work_interfere","no_employees",
        "remote_work","tech_company","benefits",
        "care_options","wellness_program","seek_help",
        "anonymity","leave","mental_health_consequence",
        "phys_health_consequence","coworkers","supervisor",
        "mental_health_interview","phys_health_interview",
        "mental_vs_physical","obs_consequence"
]


# Netegem i escalem
X = df[cols].dropna()
X_scaled = MinMaxScaler().fit_transform(X.values)

# Compute Laplacian Scores
scores = laplacian_score(X_scaled)

# Crear DataFrame per ordenar resultats
df_scores = pd.DataFrame({
    "variable": cols,
    "laplacian_score": scores
}).sort_values("laplacian_score", ascending=True)  # més baix = millor



# ================================
#          GRÀFICA FINAL
# ================================
plt.figure(figsize=(10, 6))
plt.barh(df_scores["variable"], df_scores["laplacian_score"], color="skyblue")
plt.xlabel("Laplacian Score (menor = millor)")
plt.ylabel("Variables")
plt.title("Rànquing de Variables segons Laplacian Score")
plt.gca().invert_yaxis()  # La millor variable dalt de tot
plt.tight_layout()
plt.show()

# MOSTRAR RESULTATS PER CONSOLA
print("\n===== RÀNQUING FINAL DE VARIABLES (Laplacian Score) =====")

"""


# ================================ Metode colze ================================
# Selecció de les 7 millors variables segons Laplacian Score   
top7 = [
    'treatment', 'family_history', 'remote_work',
    'mental_health_interview', 'mental_health_consequence',
    'seek_help', 'obs_consequence'
]

# ====== CARREGA I PREPARA EL DATASET ======
df = pd.read_csv("treballadors_definitiu.csv")

# Elimina files incompletes (opcional però recomanat)
df_clean = df[top7].dropna()

# Escalat Min-Max
scaler = MinMaxScaler()
X = scaler.fit_transform(df_clean)

# ====== MÈTODE DEL COLZE ======
K_RANGE = range(1, 11)
inertia_values = []

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# ====== GRÀFICA DEL COLZE ======
plt.figure(figsize=(10, 6))
plt.plot(K_RANGE, inertia_values, marker='o')
plt.title("Mètode del colze — TOP 7 variables seleccionades")
plt.xlabel("Nombre de clústers (K)")
plt.ylabel("Inèrcia (Suma de distàncies internes)")
plt.xticks(K_RANGE)
plt.grid(True)
plt.show()

# Mostrar valors per consola
for k, inertia in zip(K_RANGE, inertia_values):
    print(f"K={k}:  inertia={inertia:.3f}")


