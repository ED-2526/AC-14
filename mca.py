import pandas as pd
import prince
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# 1. Carregar dataset
# ===========================
df = pd.read_csv("treballadors_definitiu.csv")

vars_mca = [
    "work_interfere", "family_history", "no_employees", "care_options",
    "benefits", "supervisor", "treatment", "remote_work", "seek_help",
    "wellness_program", "Gender", "phys_health_consequence",
    "mental_health_consequence", "tech_company"
]

df_mca = df[vars_mca].copy().fillna("Unknown").astype(str)

print("Variables utilitzades:", df_mca.columns.tolist())

# ===========================
# 2. Executar MCA
# ===========================
mca = prince.MCA(
    n_components=10,
    random_state=42
)

mca = mca.fit(df_mca)
X_mca = mca.transform(df_mca)

print("\nShape MCA:", X_mca.shape)

# ===========================
# 3. Selecció de K amb el mètode del colze + silhouette
# ===========================
inertia = []
silhouette = []

K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_mca.iloc[:, :5])  # només primeres 5 components (les fortes)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_mca.iloc[:, :5], labels))

plt.figure(figsize=(10,4))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow method (MCA + K-Means)")
plt.xlabel("Clusters (k)")
plt.ylabel("Inertia")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(K_range, silhouette, marker="o")
plt.title("Silhouette score")
plt.xlabel("Clusters (k)")
plt.ylabel("Score")
plt.show()

# ===========================
# 4. Aplicar K-Means final
# ===========================
k_opt = 3  # CANVIA segons els gràfics
kmeans = KMeans(n_clusters=k_opt, random_state=42)
df["cluster"] = kmeans.fit_predict(X_mca.iloc[:, :5])

print("\nMida de cada cluster:")
print(df["cluster"].value_counts())

# ===========================
# 5. Visualització 2D
# ===========================
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=X_mca[0], y=X_mca[1],
    hue=df["cluster"], palette="Set2", s=60
)
plt.title("Visualització MCA (components 1 i 2)")
plt.show()

# ===========================
# 6. Perfils de clusters
# ===========================
perfil = df.groupby("cluster")[vars_mca].agg(lambda x: x.mode()[0])
print("\nPerfils categòrics dels clusters:")
print(perfil)