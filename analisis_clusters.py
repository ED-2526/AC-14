import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns

# ============================
# 1. Carregar el dataset
# ============================
df = pd.read_csv("treballadors_definitiu.csv")

# Variables seleccionades prèviament (sense NaNs i sense soroll)
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
# 2. Escalar les dades
# ============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_selected.values)

# ============================
# 3. Aplicar K-Means
# ============================
kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)

# Afegim les etiquetes de clúster a les dades
df_selected["cluster"] = labels

# ============================
# 4. Càlcul de la distància de cada punt al seu centroid
# ============================
centroids = kmeans.cluster_centers_

# Calcular distàncies de cada punt respecte al seu centroid
distances = euclidean_distances(X_scaled, centroids)

# Afegim la distància al centroid per cada treballador
df_selected["dist_to_centroid"] = distances[np.arange(len(distances)), labels]

# ============================
# 5. Visualització de la distància al centroid per clúster
# ============================
plt.figure(figsize=(12, 6))
sns.boxplot(x="cluster", y="dist_to_centroid", data=df_selected)
plt.title("Distribució de la distància al centroid per clúster")
plt.xlabel("Clúster")
plt.ylabel("Distància al centroid")
plt.show()

# ============================
# 6. Anàlisi detallada de les variables per clúster
# ============================
for var in vars_selected:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="cluster", y=var, data=df_selected)
    plt.title(f"Distribució de {var} per clúster")
    plt.xlabel("Clúster")
    plt.ylabel(var)
    plt.show()

# ============================
# 7. Anàlisi de les distàncies per cada clúster
# ============================
# Aquí veurem la mitjana de distància per clúster i per variable.
cluster_distances = df_selected.groupby("cluster")["dist_to_centroid"].mean().reset_index()

print("Distància mitjana al centroid per clúster:")
print(cluster_distances)

# ============================
# 8. Anàlisi per variables
# Aquí calculem quines variables tenen més pes en les distàncies
# i quines distàncies poden afectar més a l'assignació de clúster
# ============================
correlation_matrix = df_selected[vars_selected + ["dist_to_centroid"]].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlació entre variables i distància al centroid")
plt.show()

# ============================
# 9. Identificar treballadors que estan "fora de lloc"
# Si la distància al centroid és molt alta, aquests treballadors podrien ser interessants per estudiar
# ============================
outliers = df_selected[df_selected["dist_to_centroid"] > df_selected["dist_to_centroid"].quantile(0.75)]
print("\nTreballadors amb distància més alta al centroid (fora de lloc):")
print(outliers)
