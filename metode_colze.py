import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from kmodes.kmodes import KModes

# ================================
# 1. Carregar dataset
# ================================
df = pd.read_csv('treballadors_definitiu.csv')

# 2. Variables que farem servir per al clustering (categòriques)
features = [
    'work_interfere',
    'family_history',
    'no_employees',
    'care_options',
    'benefits',
    'supervisor',
    'treatment', 
    'Age'
]

X = df[features].copy()

print(f"Dataframe de clustering: {X.shape[0]} files, {X.shape[1]} columnes")
print("Percentatge de NaN per columna:\n", X.isna().mean())

# ================================
# 2. Imputar valors perduts
#   -> com que són categories, usem el valor més freqüent
# ================================
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

# Convertim a strings perquè KModes tracti tot com a categories
X_cat = X_imputed.astype(str)
"""
# ================================
# 3. “Mètode del codo” per K-Modes
#    (usem el cost del model com a analog a la inèrcia)
# ================================
costs = []
K = range(2, 11)

for k in K:
    km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0, random_state=42)
    km.fit(X_cat)
    costs.append(km.cost_)
    print(f"k={k}, cost={km.cost_}")

plt.figure(figsize=(8, 6))
plt.plot(K, costs, marker='o')
plt.title('Mètode del codo per K-Modes')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Cost (KModes.cost_)')
plt.xticks(K)
plt.grid(True)
plt.show()
"""
# Després de mirar el gràfic, tria el k òptim:
k_opt = 5  # <-- CANVIA A MÀ segons el codo que vegis

# ================================
# 4. Clustering final amb K-Modes
# ================================
km_final = KModes(n_clusters=k_opt, init='Huang', n_init=10, verbose=0, random_state=42)
clusters = km_final.fit_predict(X_cat)

df['cluster'] = clusters

print("\nMida de cada cluster:")
print(df['cluster'].value_counts())

# ================================
# 5. “Perfil” de cada cluster (moda de cada variable)
# ================================
def moda(s):
    return s.value_counts().idxmax()

cluster_profiles = df.groupby('cluster')[features].agg(moda)
print("\nPerfil categòric de cada cluster (moda per variable):")
print(cluster_profiles)

import matplotlib.pyplot as plt
import seaborn as sns

# Convertir els valors categòrics (strings) a números per poder fer el heatmap
cluster_numeric = cluster_profiles.copy()

for col in cluster_numeric.columns:
    try:
        cluster_numeric[col] = cluster_numeric[col].astype(float)
    except:
        pass

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_numeric, annot=True, cmap='Spectral', linewidths=0.5)
plt.title("Heatmap de perfils dels clusters (K-Modes)", fontsize=14)
plt.xlabel("Variables", fontsize=12)
plt.ylabel("Cluster", fontsize=12)
plt.tight_layout()
plt.show()
