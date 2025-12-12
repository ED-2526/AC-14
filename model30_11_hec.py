import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ======================================
# 1. LLEGIR DATASET
# ======================================
df = pd.read_csv("treballadors_definitiu.csv")
print("Dataset carregat:", df.shape)

# ======================================
# 2. ELIMINAR COLUMNES INÚTILS
# ======================================
cols_drop = ["Timestamp", "Country", "state", "comments"]
df = df.drop(columns=cols_drop, errors="ignore")

# ======================================
# 3. SEPARAR VARIABLES NUMÈRIQUES I CATEGÒRIQUES
# ======================================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("Numèriques:", num_cols)
print("Categòriques:", cat_cols)

# ======================================
# 4. PIPELINES D’IMPUTACIÓ + CODIFICACIÓ
# ======================================

# Imputar numèriques amb mitjana + escalar
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler())
])

# Imputar categòriques amb moda + OneHotEncoder
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first"))
])

# Combinar pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ======================================
# 5. PREPROCESS + CLUSTERING
# ======================================
print("Processant dades...")

X_processed = preprocessor.fit_transform(df)

print("Dades processades:", X_processed.shape)

# ======================================
# 6. K-MEANS
# ======================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_processed)

print("Clusters creats:", np.unique(clusters))

# Afegir al dataframe original
df["cluster"] = clusters

# ======================================
# 7. PCA PER VISUALITZACIÓ
# ======================================
pca = PCA(n_components=2)
coords = pca.fit_transform(X_processed)

df["PC1"] = coords[:, 0]
df["PC2"] = coords[:, 1]

print("Variància explicada PCA:", pca.explained_variance_ratio_)

# ======================================
# 8. PLOT PCA (Clusters)
# ======================================
plt.figure(figsize=(10, 7))
plt.scatter(df["PC1"], df["PC2"], c=df["cluster"], cmap="viridis", alpha=0.7)
plt.title("Clustering — PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()

# ======================================
# 9. PLOT PCA (variable treatment)
# ======================================
if "treatment" in df.columns:
    plt.figure(figsize=(10, 7))
    plt.scatter(df["PC1"], df["PC2"], c=df["treatment"], cmap="coolwarm", alpha=0.7)
    plt.title("PCA — Necessitat de Tractament Psicològic")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    cbar = plt.colorbar()
    cbar.set_label("treatment (0=No,1=Sí,2=NS/NC)")
    plt.show()
