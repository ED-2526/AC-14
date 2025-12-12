import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# ==============================
# 1. LLEGIR DATASET
# ==============================
df = pd.read_csv("treballadors_definitiu.csv")

print("Dataset carregat:", df.shape)


# ==============================
# 2. ELIMINAR COLUMNES NO ÚTILS
# ==============================
cols_a_treure = ["Timestamp", "Country", "state", "comments"]
df = df.drop(columns=cols_a_treure, errors='ignore')


# ==============================
# 3. CONVERTIR A NUMÈRIC
# ==============================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# ==============================
# 4. IMPUTAR NULS (mitjana)
# ==============================
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# ==============================
# 5. ESCALAR
# ==============================
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)


# ==============================
# 6. K-MEANS 
# ==============================
kmeans = KMeans(n_clusters=10, random_state=42)
df_scaled["cluster"] = kmeans.fit_predict(df_scaled)


# ==============================
# 7. PCA PER VISUALITZACIÓ
# ==============================
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(df_scaled.drop(columns=["cluster"]))

df_scaled["PC1"] = pca_coords[:, 0]
df_scaled["PC2"] = pca_coords[:, 1]


# ==============================
# 8. PLOT PCA — COLOR PER CLÚSTER
# ==============================
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_scaled["PC1"], df_scaled["PC2"],
                      c=df_scaled["cluster"], cmap="tab10", alpha=0.7)

plt.title("Clustering Treballadors — Totes les variables (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Cluster")
plt.show()

# ==============================
# 9. PLOT PCA — COLOR PER TREATMENT (assistència psicològica)
# ==============================
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_scaled["PC1"], df_scaled["PC2"],
                      c=df_imputed["treatment"], cmap="coolwarm", alpha=0.7)

plt.title("Treballadors — Necessitat de tractament psicològic (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
cbar = plt.colorbar(scatter)
cbar.set_label("treatment (0 = No, 1 = Sí, 2 = NS/NC)")
plt.show()
