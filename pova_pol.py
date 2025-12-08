import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.feature_selection import f_classif

import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("Set1")

# ============================================================
# 1. CARREGAR EL DATASET NETEJAT
# ============================================================
try:
    df_treballadors = pd.read_csv('treballadors_definitiu.csv')
    print("✅ Dataset de Treballadors definitiu carregat.")
except FileNotFoundError:
    print("❌ Error: No es troba 'treballadors_definitiu.csv'. Assegura't d'executar la preparació de dades.")
    raise SystemExit

# ============================================================
# 2. DEFINICIÓ DE VARIABLES I PREPARACIÓ
#    (AQUÍ ANIRÀS FENT PROVES DE QUINES VARIABLES INCLUIR)
# ============================================================

# Variables que considerem "contínues" o ordinals i volem escalar
variables_to_scale = ['Age', 'work_interfere', 'no_employees']

# Variables binàries/categòriques que deixem sense escalar
variables_no_scale = ['treatment', 'mental_vs_physical']

# Llista final de variables que entren al clustering
variables_per_clustering = variables_to_scale + variables_no_scale

# Subset de dades (eliminem files amb NaN a aquestes variables)
X_original = df_treballadors[variables_per_clustering].copy().dropna()
X = X_original.copy()

print(f"\nVariables utilitzades per al clustering: {variables_per_clustering}")
print(f"Nombre de registres (sense NaN en aquestes variables): {len(X)}")

# ============================================================
# 3. ESCALAT SELECTIU (NOMÉS PER A LES CONTÍNUES)
# ============================================================

scaler = MinMaxScaler()
X_scaled_cont = scaler.fit_transform(X[variables_to_scale])

X_scaled_df = pd.DataFrame(
    X_scaled_cont,
    columns=variables_to_scale,
    index=X.index
)

# Combina variables escalades + variables sense escalar
X_final = pd.concat([X_scaled_df, X[variables_no_scale]], axis=1)
X_final_array = X_final.values

print(f"\nDades finals preparades (MinMax per {len(variables_to_scale)} variables): {len(X_final)} registres.")

# ============================================================
# 4. APLICACIÓ DE DIFERENTS MODELS DE CLUSTERING
# ============================================================

K = 8   # 👉 Canvia aquest valor per provar diferents nombres de clústers
models_info = []

# --- K-Means ---
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_final_array)
df_treballadors.loc[X_final.index, 'cluster_kmeans'] = kmeans_labels.astype(int)
models_info.append(('K-Means', kmeans_labels, 'cluster_kmeans'))

# --- Gaussian Mixture Model (GMM) ---
gmm = GaussianMixture(n_components=K, random_state=42)
gmm_labels = gmm.fit_predict(X_final_array)
df_treballadors.loc[X_final.index, 'cluster_gmm'] = gmm_labels.astype(int)
models_info.append(('Gaussian Mixture', gmm_labels, 'cluster_gmm'))

# --- Clustering Jeràrquic (Agglomerative, Ward) ---
hierarchical = AgglomerativeClustering(n_clusters=K, linkage='ward')
hier_labels = hierarchical.fit_predict(X_final_array)
df_treballadors.loc[X_final.index, 'cluster_hierarchical'] = hier_labels.astype(int)
models_info.append(('Jeràrquic (Ward)', hier_labels, 'cluster_hierarchical'))

print(f"\n✅ Models aplicats per K={K}: K-Means, Gaussian Mixture, Clustering Jeràrquic")

# ============================================================
# 5. MÈTRIQUES GLOBALS PER MODEL (DBS, BSS/TSS, SILHOUETTE)
# ============================================================

def calcular_bss_tss(X_np, clusters):
    """
    Retorna el ratio BSS/TSS (entre 0 i 1).
    Com més a prop d'1, millor separació entre clústers.
    """
    centroide_global = X_np.mean(axis=0)
    TSS = np.sum(np.sum((X_np - centroide_global) ** 2, axis=1))
    BSS = 0.0
    for c in np.unique(clusters):
        points_c = X_np[clusters == c]
        if len(points_c) == 0:
            continue
        centroide_c = points_c.mean(axis=0)
        BSS += len(points_c) * np.sum((centroide_c - centroide_global) ** 2)
    return BSS / TSS if TSS > 0 else np.nan

print("\n================ MÈTRIQUES PER MODEL ================")
for nom_model, labels, colname in models_info:
    try:
        dbs = davies_bouldin_score(X_final_array, labels)
    except Exception:
        dbs = np.nan

    try:
        sil = silhouette_score(X_final_array, labels)
    except Exception:
        sil = np.nan

    ratio = calcular_bss_tss(X_final_array, labels)

    print(f"\n{nom_model} (columna='{colname}'):")
    print(f"  - Davies-Bouldin Score (↓ millor): {dbs:.3f}")
    print(f"  - Silhouette Score (↑ millor):    {sil:.3f}")
    print(f"  - BSS/TSS (↑ millor):            {ratio:.3f}")
print("=====================================================\n")

# Per a la resta de l'anàlisi ens quedem amb K-Means com a model base
cluster_col = 'cluster_kmeans'
labels = kmeans_labels

# ============================================================
# 6. IMPORTÀNCIA DE VARIABLES (QUINES DIFERENCIEN MÉS ELS CLÚSTERS?)
#    Usem F-test (ANOVA) respecte als clústers de K-Means
# ============================================================

f_vals, p_vals = f_classif(X_final, labels)
importancia = pd.DataFrame({
    'variable': X_final.columns,
    'F_value': f_vals,
    'p_value': p_vals
}).sort_values('p_value')

print("📊 IMPORTÀNCIA DE LES VARIABLES (respecte els clústers K-Means)")
print(importancia.to_string(index=False))

print("""
Interpretació ràpida:
  - p_value petit (p < 0.05): la variable CANVIA bastant entre clústers → és rellevant.
  - p_value gran      (p > 0.05): la variable gairebé no canvia entre clústers → potser poc significativa.
""")

# ============================================================
# 7. RESUM NUMÈRIC PER CLÚSTER (K-MEANS)
# ============================================================

# Dades originals + etiqueta de clúster
df_cl = df_treballadors.loc[X_final.index, variables_per_clustering + [cluster_col]].copy()

cluster_summary = df_cl.groupby(cluster_col).agg(
    Poblacio=('Age', 'size'),
    Age_Mitjana=('Age', 'mean'),
    Interferencia_Mitjana=('work_interfere', 'mean'),
    Emp_Mitjana=('no_employees', 'mean'),
    Percent_Tractament=('treatment', 'mean'),
    Percent_Igualtat_MTL_PHY=('mental_vs_physical', 'mean')
).reset_index()

# Renombrem la columna de clúster perquè quedi maco als gràfics
cluster_summary = cluster_summary.rename(columns={cluster_col: 'Cluster'})

# Percentatges
cluster_summary['Percent_Tractament'] = (cluster_summary['Percent_Tractament'] * 100).round(1)
cluster_summary['Percent_Igualtat_MTL_PHY'] = (cluster_summary['Percent_Igualtat_MTL_PHY'] * 100).round(1)

print("\nRESUM NUMÈRIC PER CLÚSTER (K-Means)")
print("---------------------------------------")
print(cluster_summary.to_string(index=False))
print("\nNota:")
print("  - 'Interferencia_Mitjana' (1 = Gairebé mai, 4 = Molta interferència).")
print("  - 'Emp_Mitjana' (1 = Empresa petita, 6 = Empresa molt gran).")

# ============================================================
# 8. VISUALITZACIÓ PER INTERPRETAR ELS CLÚSTERS (K-MEANS)
# ============================================================

plt.figure(figsize=(18, 12))

# Subplot 1: Impacte laboral (work_interfere)
plt.subplot(2, 2, 1)
sns.barplot(x='Cluster', y='Interferencia_Mitjana', data=cluster_summary, palette='viridis')
plt.title("Grau d'interferència laboral (work_interfere) per clúster", fontsize=14)
plt.ylabel('Mitjana (1 = Gairebé mai, 4 = Molta)')
plt.ylim(0, cluster_summary['Interferencia_Mitjana'].max() * 1.2)
for _, row in cluster_summary.iterrows():
    plt.text(row.Cluster, row.Interferencia_Mitjana,
             f"{row.Interferencia_Mitjana:.2f}",
             ha="center", va="bottom", fontweight='bold')

# Subplot 2: Percentatge amb tractament
plt.subplot(2, 2, 2)
sns.barplot(x='Cluster', y='Percent_Tractament', data=cluster_summary, palette='magma')
plt.title('Percentatge de treballadors amb tractament per clúster', fontsize=14)
plt.ylabel('Percentatge (%)')
plt.ylim(0, 105)
for _, row in cluster_summary.iterrows():
    plt.text(row.Cluster, row.Percent_Tractament + 2,
             f"{row.Percent_Tractament:.1f}%",
             ha="center", va="bottom", fontweight='bold')

# Subplot 3: Edat mitjana
plt.subplot(2, 2, 3)
sns.barplot(x='Cluster', y='Age_Mitjana', data=cluster_summary, palette='plasma')
plt.title('Edat mitjana per clúster', fontsize=14)
plt.ylabel('Anys')
plt.ylim(0, cluster_summary['Age_Mitjana'].max() * 1.1)
for _, row in cluster_summary.iterrows():
    plt.text(row.Cluster, row.Age_Mitjana,
             f"{row.Age_Mitjana:.1f}",
             ha="center", va="bottom", fontweight='bold')

# Subplot 4: Mida de l'empresa
plt.subplot(2, 2, 4)
sns.barplot(x='Cluster', y='Emp_Mitjana', data=cluster_summary, palette='cool')
plt.title("Mida mitjana de l'empresa (no_employees) per clúster", fontsize=14)
plt.ylabel('Mitjana (1 = Petita, 6 = Molt gran)')
plt.ylim(0, cluster_summary['Emp_Mitjana'].max() * 1.2)
for _, row in cluster_summary.iterrows():
    plt.text(row.Cluster, row.Emp_Mitjana,
             f"{row.Emp_Mitjana:.2f}",
             ha="center", va="bottom", fontweight='bold')

plt.suptitle(f'INTERPRETACIÓ DELS {K} CLÚSTERS (K-Means)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nTorna a executar amb diferents K o diferents llistes de variables per anar provant configuracions.")
