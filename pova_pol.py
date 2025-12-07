import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("Set1")

# 1. CARREGAR EL DATASET NETEJAT
try:
    df_treballadors = pd.read_csv('treballadors_definitiu.csv')
    print("Dataset de Treballadors definitiu carregat.")
except FileNotFoundError:
    print("Error: No es troba 'treballadors_definitiu.csv'. Assegura't d'executar la preparació de dades.")
    exit()

# 2. DEFINICIÓ DE VARIABLES I PREPARACIÓ (Configuració Òptima)
variables_to_scale = ['Age', 'work_interfere', 'no_employees'] 
variables_no_scale = ['treatment', 'mental_vs_physical'] 
variables_per_clustering = variables_to_scale + variables_no_scale

X_original = df_treballadors[variables_per_clustering].copy().dropna()
X = X_original.copy()

# 3. APLICACIÓ DE L'ESCALAT SELECTIU (MINMAXSCALER)
scaler = MinMaxScaler()
X_scaled_parts = scaler.fit_transform(X[variables_to_scale])
X_scaled_df = pd.DataFrame(X_scaled_parts, columns=variables_to_scale, index=X.index)
# Per al clustering, utilitzem les variables escalades + les binàries (treatment, mental_vs_physical) sense escalar
X_final = pd.concat([X_scaled_df, X[variables_no_scale]], axis=1)
X_final_array = X_final.values

print(f"\nDades finals preparades (MinMax Scaling selectiva, 5 variables): {len(X_final)} registres.")

# ====================================================================
# 4. ENTRENAMENT DEL MODEL K-MEANS FINAL (K=4)
# ====================================================================

K = 8 
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(X_final_array)

# Assignem els clústers al DataFrame original SENSE escalar per a la interpretació
X_original['Cluster'] = kmeans_clusters

print(f"\nModel K-Means final entrenat amb K={K}. Clusters assignats per a l'anàlisi.")
print("====================================================================")

# Càlcul de mètriques per a la presentació final
def calcular_bss_tss(X, clusters):
    centroide_global = X.mean(axis=0)
    TSS = np.sum(np.sum((X - centroide_global)**2, axis=1))
    BSS = 0
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        cluster_points = X[clusters == cluster_id]
        if len(cluster_points) > 0:
            centroide_cluster = cluster_points.mean(axis=0)
            BSS += len(cluster_points) * np.sum((centroide_cluster - centroide_global)**2)
    return BSS / TSS if TSS > 0 else 0

try:
    dbs_final = davies_bouldin_score(X_final_array, kmeans_clusters)
except Exception:
    dbs_final = np.nan
ratio_final = calcular_bss_tss(X_final_array, kmeans_clusters)

print(f"Mètriques del Model K-Means Final (K={K}):")
print(f"  Davies-Bouldin Score (DBS): {dbs_final:.3f} (Objectiu: Min. 0)")
print(f"  Ratio BSS/TSS: {ratio_final:.3f} (Objectiu: Max. 1)")
print("====================================================================")


# ====================================================================
# 5. ANÀLISI DE CLÚSTERS PER A LA INTERPRETACIÓ
# ====================================================================

# Calculem la mitjana (o el recompte) de cada variable per clúster
cluster_summary = X_original.groupby('Cluster').agg(
    Poblacio=('Cluster', 'size'),
    Age_Mitjana=('Age', 'mean'),
    Interferencia_Mitjana=('work_interfere', 'mean'),
    Emp_Mitjana=('no_employees', 'mean'),
    Percent_Tractament=('treatment', 'mean'),
    Percent_Igualtat_MTL_PHY=('mental_vs_physical', 'mean')
).reset_index()

# Convertim les mitjanes a percentatges
cluster_summary['Percent_Tractament'] = (cluster_summary['Percent_Tractament'] * 100).round(1)
cluster_summary['Percent_Igualtat_MTL_PHY'] = (cluster_summary['Percent_Igualtat_MTL_PHY'] * 100).round(1)

print("\nRESUM NUMÈRIC PER CLÚSTER (Interpretació dels 4 Grups)")
print("---------------------------------------")
print(cluster_summary.to_string(index=False))
print("\nNota: 'Interferencia_Mitjana' (1=Poc, 4=Molta). 'Emp_Mitjana' (1=Petita, 6=Molt Gran).")

# ====================================================================
# 6. VISUALITZACIÓ DE LA INTERPRETACIÓ
# ====================================================================

plt.figure(figsize=(18, 12))

# Subplot 1: Impacte Laboral (work_interfere)
plt.subplot(2, 2, 1)
sns.barplot(x='Cluster', y='Interferencia_Mitjana', data=cluster_summary, palette='viridis')
plt.title('Grau d\'Interferència Laboral (work_interfere) per Clúster', fontsize=14)
plt.ylabel('Mitjana (1=Gairebé Mai, 4=Molta)')
plt.ylim(0, cluster_summary['Interferencia_Mitjana'].max() * 1.2)
for index, row in cluster_summary.iterrows():
    plt.text(row.Cluster, row.Interferencia_Mitjana, f"{row.Interferencia_Mitjana:.2f}", color='black', ha="center", va="bottom", fontweight='bold')

# Subplot 2: Percentatge amb Tractament (treatment)
plt.subplot(2, 2, 2)
sns.barplot(x='Cluster', y='Percent_Tractament', data=cluster_summary, palette='magma')
plt.title('Percentatge de Treballadors amb Tractament per Clúster', fontsize=14)
plt.ylabel('Percentatge (%)')
plt.ylim(0, 105)
for index, row in cluster_summary.iterrows():
    plt.text(row.Cluster, row.Percent_Tractament + 2, f"{row.Percent_Tractament:.1f}%", color='black', ha="center", va="bottom", fontweight='bold')

# Subplot 3: Mitjana d'Edat
plt.subplot(2, 2, 3)
sns.barplot(x='Cluster', y='Age_Mitjana', data=cluster_summary, palette='plasma')
plt.title('Edat Mitjana per Clúster', fontsize=14)
plt.ylabel('Anys')
plt.ylim(0, cluster_summary['Age_Mitjana'].max() * 1.1)
for index, row in cluster_summary.iterrows():
    plt.text(row.Cluster, row.Age_Mitjana, f"{row.Age_Mitjana:.1f}", color='black', ha="center", va="bottom", fontweight='bold')

# Subplot 4: Mida de l'Empresa (no_employees)
plt.subplot(2, 2, 4)
sns.barplot(x='Cluster', y='Emp_Mitjana', data=cluster_summary, palette='cool')
plt.title('Mida Mitjana de l\'Empresa (no_employees) per Clúster', fontsize=14)
plt.ylabel('Mitjana (1=Petita, 6=Molt Gran)')
plt.ylim(0, cluster_summary['Emp_Mitjana'].max() * 1.2)
for index, row in cluster_summary.iterrows():
    plt.text(row.Cluster, row.Emp_Mitjana, f"{row.Emp_Mitjana:.2f}", color='black', ha="center", va="bottom", fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f'INTERPRETACIÓ DELS {K} CLÚSTERS K-MEANS', fontsize=16, fontweight='bold')
plt.show()

print("\nEXECUTA EL CODI DE NOU per visualitzar el resum i els gràfics d'interpretació per a cadascun dels 4 clústers. Amb aquesta informació ja podràs etiquetar els grups.")