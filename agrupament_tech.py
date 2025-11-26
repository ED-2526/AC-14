import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

df_combinat = pd.read_csv('dataset_combinado.csv')
print("Dataset combinat carregat")

df_tech = df_combinat[
    (df_combinat['source'] == 2) & 
    (df_combinat['tech_company'] == 1)
].copy()

df_tech_clean = df_tech[df_tech['age'] <= 100].copy()

variables_binaries = ['treatment', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity']
for var in variables_binaries:
    df_tech_clean[var] = df_tech_clean[var].replace(2, 0)

variables_per_clustering = [
    'age', 'sex', 'treatment', 'work_interfere', 'benefits', 
    'care_options', 'wellness_program', 'seek_help', 'anonymity'
]

X = df_tech_clean[variables_per_clustering].dropna()

scaler_standard = StandardScaler()
X_scaled = scaler_standard.fit_transform(X)

print(f"Dades preparades: {len(X)} registres")

print("APLICANT MODELS DE CLUSTERING")

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(X_scaled)
df_tech_clean.loc[X.index, 'cluster_kmeans'] = kmeans_clusters

gmm = GaussianMixture(n_components=4, random_state=42)
gmm_clusters = gmm.fit_predict(X_scaled)
df_tech_clean.loc[X.index, 'cluster_gmm'] = gmm_clusters

hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
hierarchical_clusters = hierarchical.fit_predict(X_scaled)
df_tech_clean.loc[X.index, 'cluster_hierarchical'] = hierarchical_clusters

print("Models aplicats: K-Means, Gaussian Mixture, Clustering Jerarquic")

print("CALCUL DE METRIQUES BSS I TSS")

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
    
    return BSS, TSS, SSE, ratio_bss_tss

print("METRIQUES D'AVALUACIO:")
print("=" * 50)

kmeans_bss, kmeans_tss, kmeans_sse, kmeans_ratio = calcular_bss_tss(X_scaled, kmeans_clusters, "K-Means")
gmm_bss, gmm_tss, gmm_sse, gmm_ratio = calcular_bss_tss(X_scaled, gmm_clusters, "Gaussian Mixture")
hierarchical_bss, hierarchical_tss, hierarchical_sse, hierarchical_ratio = calcular_bss_tss(X_scaled, hierarchical_clusters, "Jerarquic")

print("Generant visualitzacions...")

# PRIMERA FIGURA: GRÁFICOS ORIGINALES + NUEVOS DE BSS/TSS
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('COMPARACIO DE MODELS DE CLUSTERING', fontsize=16, fontweight='bold')

# Datos para los gráficos
models = ['K-Means', 'Gaussian Mixture', 'Jerarquic']
bss_values = [kmeans_bss, gmm_bss, hierarchical_bss]
tss_values = [kmeans_tss, gmm_tss, hierarchical_tss]
sse_values = [kmeans_sse, gmm_sse, hierarchical_sse]
ratios = [kmeans_ratio, gmm_ratio, hierarchical_ratio]

# Gráfico 1: SSE (Original)
axes[0, 0].bar(models, sse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0, 0].set_title('SSE (Sum of Squared Errors)')
axes[0, 0].set_ylabel('SSE')
for i, v in enumerate(sse_values):
    axes[0, 0].text(i, v + 50, f'{v:.0f}', ha='center', va='bottom')

# Gráfico 2: Ratio BSS/TSS (Original)
axes[0, 1].bar(models, ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0, 1].set_title('Ratio BSS/TSS')
axes[0, 1].set_ylabel('Ratio')
axes[0, 1].set_ylim(0, 1)
for i, v in enumerate(ratios):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# Gráfico 3: Distribución de clusters (Original)
for i, (model, columna) in enumerate([('K-Means', 'cluster_kmeans'), 
                                    ('Gaussian Mixture', 'cluster_gmm'),
                                    ('Jerarquic', 'cluster_hierarchical')]):
    counts = df_tech_clean[columna].value_counts().sort_index()
    axes[1, 0].bar(np.arange(len(counts)) + i*0.25, counts.values, width=0.25, 
                   label=model, alpha=0.7)

axes[1, 0].set_title('Distribucio de Clusters')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Nombre de Persones')
axes[1, 0].legend()

# Gráfico 4: Tratamiento por cluster (Original)
models_info = [
    ('K-Means', 'cluster_kmeans'),
    ('Gaussian Mixture', 'cluster_gmm'), 
    ('Jerarquic', 'cluster_hierarchical')
]

treatment_data = []
for nom, columna in models_info:
    treatment_rates = df_tech_clean.groupby(columna)['treatment'].mean() * 100
    treatment_data.append((nom, treatment_rates))

x_pos = np.arange(4)
width = 0.25

for i, (nom, treatment_rates) in enumerate(treatment_data):
    axes[1, 1].bar(x_pos + i * width, treatment_rates.values, width, label=nom, alpha=0.7)

axes[1, 1].set_title('Tractament per Cluster i Model')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Percentatge amb Tractament (%)')
axes[1, 1].set_xticks(x_pos + width)
axes[1, 1].set_xticklabels(['0', '1', '2', '3'])
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# SEGUNDA FIGURA: GRÁFICOS ESPECÍFICOS DE BSS Y TSS
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ANÁLISIS DETALLADO DE BSS Y TSS', fontsize=16, fontweight='bold')

# Gráfico 1: BSS por modelo
bars1 = axes[0, 0].bar(models, bss_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
axes[0, 0].set_title('BSS (Between Sum of Squares) per Model')
axes[0, 0].set_ylabel('BSS')
for bar, value in zip(bars1, bss_values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

# Gráfico 2: TSS por modelo
bars2 = axes[0, 1].bar(models, tss_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
axes[0, 1].set_title('TSS (Total Sum of Squares) per Model')
axes[0, 1].set_ylabel('TSS')
for bar, value in zip(bars2, tss_values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

# Gráfico 3: Comparación BSS vs TSS
x_pos = np.arange(len(models))
width = 0.35

bars_bss = axes[1, 0].bar(x_pos - width/2, bss_values, width, label='BSS', color='blue', alpha=0.7)
bars_tss = axes[1, 0].bar(x_pos + width/2, tss_values, width, label='TSS', color='red', alpha=0.7)
axes[1, 0].set_title('Comparació BSS vs TSS')
axes[1, 0].set_ylabel('Valor')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(models)
axes[1, 0].legend()

# Añadir valores en las barras
for bar, value in zip(bars_bss, bss_values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
for bar, value in zip(bars_tss, tss_values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)

# Gráfico 4: Ratio BSS/TSS detallado
bars4 = axes[1, 1].bar(models, ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
axes[1, 1].set_title('Ratio BSS/TSS (Variança Explicada)')
axes[1, 1].set_ylabel('Ratio')
axes[1, 1].set_ylim(0, 1)
for bar, value in zip(bars4, ratios):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                   f'{value*100:.1f}%', ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# TERCERA FIGURA: Dendrograma (Original)
plt.figure(figsize=(12, 8))
linked = linkage(X_scaled, 'ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, truncate_mode='lastp', p=20)
plt.title('Dendrograma - Clustering Jerarquic')
plt.xlabel('Mostres')
plt.ylabel('Distancia')
plt.show()

print("CONCLUSIO FINAL")
print("=" * 50)

print(f"TSS (comú a tots els models): {tss_values[0]:.2f}")
print("")

millor_model_sse = np.argmin(sse_values)
print(f"MILLOR MODEL PER SSE: {models[millor_model_sse]}")
print(f"SSE: {sse_values[millor_model_sse]:.2f}")

millor_model_bss = np.argmax(bss_values)
print(f"MILLOR MODEL PER BSS: {models[millor_model_bss]}")
print(f"BSS: {bss_values[millor_model_bss]:.2f}")

millor_model_ratio = np.argmax(ratios)
print(f"MILLOR MODEL PER RATIO BSS/TSS: {models[millor_model_ratio]}")
print(f"RATIO: {ratios[millor_model_ratio]:.3f}")

print("")
print("Interpretació de mètriques:")
print("- SSE (Within SS): Variància DINS dels clusters - Més baix = millor")
print("- BSS (Between SS): Variància ENTRE clusters - Més alt = millor")
print("- TSS (Total SS): Variància TOTAL - Constant per a tots els models") 
print("- Ratio BSS/TSS: Proporció de variància explicada pels clusters (0-1)")

print("Analisi completat!") 