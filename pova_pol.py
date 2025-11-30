import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
# Canvi: Importem MinMaxScaler en lloc de StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mode # Per a la funció de GMM
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("Set1")


df_treballadors = pd.read_csv('treballadors_definitiu.csv')
print("Dataset de Treballadors definitiu carregat.")


# 2. DEFINICIÓ DE VARIABLES I PREPARACIÓ
# Regla: NO estandaritzar variables binàries (0 o 1).

# Variables contínues/ordinals a escalar
# PROVA #7: Mantenim les 5 millors variables
variables_to_scale = ['Age', 'work_interfere', 'no_employees'] 
# Variables binàries a utilitzar sense escalar
variables_no_scale = ['treatment', 'mental_vs_physical'] 

# Combinació utilitzada per a la Prova #7 (amb escalat selectiu, K=4, 5 variables)
variables_per_clustering = variables_to_scale + variables_no_scale

# Eliminem NaNs per estar segurs (tot i que ja hauria d'estar net de la fase prèvia)
X_original = df_treballadors[variables_per_clustering].copy().dropna()
X = X_original.copy()

# 3. APLICACIÓ DE L'ESCALAT SELECTIU (CANVIAT A MINMAXSCALER)

# Canvi clau: usem MinMaxScaler
scaler_minmax = MinMaxScaler()
X_scaled_parts = scaler_minmax.fit_transform(X[variables_to_scale])

# Unim les parts escalades amb les parts no escalades per obtenir la matriu final (X_final)
X_scaled_df = pd.DataFrame(X_scaled_parts, columns=variables_to_scale, index=X.index)
X_final = pd.concat([X_scaled_df, X[variables_no_scale]], axis=1)

# Convertim X_final a numpy array per als algorismes
X_final_array = X_final.values



# ====================================================================
# 4. APLICACIÓ I AVALUACIÓ DELS MODELS (K=4)


K = 4 # Valor de K (prova colze)
models_info = []

# K-Means
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(X_final_array)
df_treballadors.loc[X_final.index, 'cluster_kmeans'] = kmeans_clusters.astype(int)
models_info.append(('K-Means', kmeans_clusters, 'cluster_kmeans'))

# Plot K-Means points
plt.figure(figsize=(8, 6))

# Fem servir les dues primeres columnes de X_final_array (Age i work_interfere escalats)
plt.scatter(
    X_final_array[:, 0],   # Age (escalat)
    X_final_array[:, 1],   # work_interfere (escalat)
    c=kmeans_clusters,     # color segons el cluster
    alpha=0.8
)

plt.xlabel('Age (escalat)')
plt.ylabel('Work_interfere (escalat)')
plt.title('K-Means (K=4) sobre treballadors')
plt.colorbar(label='Cluster K-Means')
plt.tight_layout()
plt.show()


# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=K, random_state=42)
gmm.fit(X_final_array)
gmm_clusters = gmm.predict(X_final_array)
df_treballadors.loc[X_final.index, 'cluster_gmm'] = gmm_clusters.astype(int)
models_info.append(('Gaussian Mixture', gmm_clusters, 'cluster_gmm'))

# Clustering Jeràrquic (Agglomerative)
hierarchical = AgglomerativeClustering(n_clusters=K, linkage='ward')
hierarchical_clusters = hierarchical.fit_predict(X_final_array)
df_treballadors.loc[X_final.index, 'cluster_hierarchical'] = hierarchical_clusters.astype(int)
models_info.append(('Jeràrquic (Ward)', hierarchical_clusters, 'cluster_hierarchical'))

print(f"\nModels aplicats per K={K}: K-Means, Gaussian Mixture, Clustering Jeràrquic")

# Funció BSS/TSS
def calcular_bss_tss(X, clusters, nom_model):
    centroide_global = X.mean(axis=0)
    TSS = np.sum(np.sum((X - centroide_global)**2, axis=1))
    
    BSS = 0
    unique_clusters = np.unique(clusters)
    
    for cluster_id in unique_clusters:
        cluster_points = X[clusters == cluster_id]
        if len(cluster_points) > 0:
            centroide_cluster = cluster_points.mean(axis=0)
            BSS += len(cluster_points) * np.sum((centroide_cluster - centroide_global)**2)
    
    SSE = TSS - BSS
    ratio_bss_tss = BSS / TSS if TSS > 0 else 0
    
    print(f"{nom_model}:")
    print(f"  BSS: {BSS:.2f}")
    print(f"  TSS: {TSS:.2f}")
    print(f"  SSE: {SSE:.2f}")
    print(f"  Ratio BSS/TSS: {ratio_bss_tss:.3f}")
    print("")
    
    return BSS, TSS, SSE, ratio_bss_tss, nom_model

print("\nMETRIQUES D'AVALUACIO:")
print("=" * 50)

results = []
for nom, clusters, _ in models_info:
    # ATENCIÓ: Tornem a calcular BSS/TSS utilitzant el nou X_final_array
    bss, tss, sse, ratio, _ = calcular_bss_tss(X_final_array, clusters, nom)
    results.append({'model': nom, 'sse': sse, 'ratio': ratio})

# Extraure dades per als gràfics
models = [r['model'] for r in results]
sse_values = [r['sse'] for r in results]
ratios = [r['ratio'] for r in results]

# ====================================================================
# 5. VISUALITZACIÓ DELS RESULTATS
# ====================================================================

# PRIMERA FIGURA: GRÀFICS DE COMPARACIÓ DE MÈTRIQUES
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'COMPARACIÓ DE MODELS DE CLUSTERING (K={K}) ', fontsize=16, fontweight='bold')

# Gráfico 1: SSE (Sum of Squared Errors)
axes[0, 0].bar(models, sse_values, color=['#A60628', '#348AA7', '#7A68A6'])
axes[0, 0].set_title('SSE (Sum of Squared Errors)')
axes[0, 0].set_ylabel('SSE')
for i, v in enumerate(sse_values):
    axes[0, 0].text(i, v + 50, f'{v:.0f}', ha='center', va='bottom')

# Gráfico 2: Ratio BSS/TSS (Objectiu)
bars_ratio = axes[0, 1].bar(models, ratios, color=['#A60628', '#348AA7', '#7A68A6'])
axes[0, 1].set_title('Ratio BSS/TSS')
axes[0, 1].set_ylabel('Ratio')
axes[0, 1].set_ylim(0, 1)
for bar, v in zip(bars_ratio, ratios):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, v + 0.02, 
                    f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, v/2, 
                    f'{v*100:.1f}%', ha='center', va='center', color='white', fontweight='bold')

# Gráfico 3: Distribución de clusters
colors = sns.color_palette("Set1", n_colors=K)
width = 0.25
x_pos = np.arange(K)

for i, (nom, _, columna) in enumerate(models_info):
    counts = df_treballadors[columna].value_counts().sort_index()
    # Assegurem-nos que tenim dades per als K clústers
    current_counts = [counts.get(c, 0) for c in x_pos]
    bars = axes[1, 0].bar(x_pos + i*width, current_counts, width=width, 
                         label=nom, alpha=0.8, color=colors[i])

axes[1, 0].set_title('Distribució de Mida dels Clusters')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Nombre de Persones')
axes[1, 0].set_xticks(x_pos + width)
axes[1, 0].set_xticklabels([f'{c}' for c in x_pos])
axes[1, 0].legend()

# Gráfico 4: Tractament per cluster (Variable Clau)
for i, (nom, _, columna) in enumerate(models_info):
    treatment_rates = df_treballadors.groupby(columna)['treatment'].mean() * 100
    current_rates = [treatment_rates.get(c, 0) for c in x_pos]
    axes[1, 1].bar(x_pos + i * width, current_rates, width, label=nom, alpha=0.8, color=colors[i])

axes[1, 1].set_title('Taxa de Tractament per Cluster i Model')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Percentatge amb Tractament (%)')
axes[1, 1].set_xticks(x_pos + width)
axes[1, 1].set_xticklabels([f'{c}' for c in x_pos])
axes[1, 1].legend(loc='upper right')

plt.tight_layout()
plt.show()

# SEGONA FIGURA: Dendrograma (Només Jeràrquic)
plt.figure(figsize=(12, 8))
# Usem X_final_array (dades escalades selectivament)
linked = linkage(X_final_array, 'ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, truncate_mode='lastp', p=20)
plt.title('Dendrograma - Clustering Jeràrquic (Prova #7, K=4)')
plt.xlabel('Mostres')
plt.ylabel('Distancia')
plt.show()

# 6. CONCLUSIÓ FINAL DE LA PROVA
print("\n" + "=" * 50)
print("RESULTATS DE LA PROVA D'OPTIMITZACIÓ")
print("Variables: Age, work_interfere, no_employees (MINMAX SCALED) + treatment, mental_vs_physical (NO SCALED)")
print("=" * 50)

# Valor de referència de la Prova #6 (K=4, Standard Scaling, 5 variables)
best_ratio_prev = 0.571 

# Trobar el millor model d'aquesta prova
best_ratio = max(ratios)
best_model_index = ratios.index(best_ratio)
best_model_name = models[best_model_index]

print(f"MILLOR MODEL PER RATIO BSS/TSS (K={K}): {best_model_name}")
print(f"RATIO OBTINGUDA: {best_ratio:.3f}")

