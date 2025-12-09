import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# =============================
# 1) VARIABLES SELECCIONADES
# =============================

top7 = [
    'treatment', 'family_history', 'remote_work',
    'mental_health_interview', 'mental_health_consequence',
    'seek_help', 'obs_consequence'
]

# =============================
# 2) CARREGAR I ESCALAR DADES
# =============================

df = pd.read_csv("treballadors_definitiu.csv")

df_clean = df[top7].dropna()

scaler = MinMaxScaler()
X = scaler.fit_transform(df_clean)

# =============================
# 3) CLUSTERING FINAL (K = 5)
# =============================

k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X)

df_clean["cluster"] = labels

print("\n===== Distribució de treballadors per clúster =====")
print(df_clean["cluster"].value_counts())

# =============================
# 4) t-SNE PER VISUALITZAR EN 2D
# =============================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1500,
    random_state=42,
    init="random"
)

embedding = tsne.fit_transform(X)

df_clean["TSNE1"] = embedding[:, 0]
df_clean["TSNE2"] = embedding[:, 1]

# =============================
# 5) GRÀFIC DELS CLÚSTERS AMB t-SNE
# =============================

plt.figure(figsize=(10, 7))

for c in range(k):
    subset = df_clean[df_clean["cluster"] == c]
    plt.scatter(subset["TSNE1"], subset["TSNE2"], label=f"Clúster {c}", alpha=0.7)

plt.title("Clustering final (K = 5) visualitzat amb t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True)
plt.show()

# =============================
# 6) PERFIL DE CADA CLÚSTER
# =============================

print("\n===== Perfil mitjà de cada clúster (TOP 7 variables) =====")
print(df_clean.groupby("cluster")[top7].mean().round(3))
