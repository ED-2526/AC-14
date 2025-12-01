import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 1. CARGA Y PREPARACION DE DATOS
# ====================================================================

print("CARGANDO Y ANALIZANDO DATOS...")

df = pd.read_csv('treballadors_definitiu.csv')
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# Eliminar filas con valores nulos en variables clave
df_clean = df.dropna()
print(f"Despues de limpiar nulos: {df_clean.shape[0]} filas")

# ====================================================================
# 2. SELECCION DE VARIABLES PARA EXPERIMENTOS
# ====================================================================

# Definir diferentes combinaciones de variables para probar
variable_combinations = {
    # Combinacion 1: Variables basicas demograficas
    "combo1_demograficas": ['Age', 'Gender', 'no_employees'],
    
    # Combinacion 2: Variables de salud mental clave
    "combo2_salud_mental": ['treatment', 'work_interfere', 'family_history', 'mental_health_consequence'],
    
    # Combinacion 3: Recursos y apoyo
    "combo3_recursos": ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity'],
    
    # Combinacion 4: Ambiente laboral
    "combo4_ambiente": ['remote_work', 'tech_company', 'leave', 'coworkers', 'supervisor'],
    
    # Combinacion 5: Mezcla optima
    "combo5_optima": ['Age', 'work_interfere', 'treatment', 'benefits', 'mental_health_consequence'],
    
    # Combinacion 6: Solo variables clave
    "combo6_minima": ['Age', 'treatment', 'work_interfere'],
    
    # NUEVA COMBINACION: Configuracion especifica solicitada
    "combo7_config_especial": ['Age', 'work_interfere', 'no_employees', 'treatment', 'mental_vs_physical']
}

print(f"COMBINACIONES DE VARIABLES A PROBAR:")
for name, vars_list in variable_combinations.items():
    print(f"  {name}: {len(vars_list)} variables")


# ====================================================================
# 3. FUNCION DE EVALUACION CON DAVIES-BOULDIN Y CALINSKI-HARABASZ
# ====================================================================

def evaluar_clustering(X, n_clusters, metodo):
    """
    Evalua clustering usando Davies-Bouldin (menor = mejor) y Calinski-Harabasz (mayor = mejor)
    """
    try:
        if metodo == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)
        elif metodo == 'gaussian':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(X)
        elif metodo == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = model.fit_predict(X)
        else:
            return None, None, None
            
        # Calcular ambas metricas
        db_score = davies_bouldin_score(X, labels)  # Menor = mejor
        ch_score = calinski_harabasz_score(X, labels)  # Mayor = mejor
        
        return db_score, ch_score, labels
        
    except Exception as e:
        return None, None, None

# ====================================================================
# 4. EXPERIMENTO PRINCIPAL: BUSQUEDA DE LA MEJOR CONFIGURACION
# ====================================================================

print("INICIANDO EXPERIMENTO DE CLUSTERING...")
print("Probando: 2-10 clusters x 3 metodos x 7 combinaciones de variables")

resultados = []
mejor_score_db_global = float('inf')  # Para Davies-Bouldin (menor es mejor)
mejor_score_ch_global = 0  # Para Calinski-Harabasz (mayor es mejor)
mejor_configuracion_db = None
mejor_configuracion_ch = None

# Probar TODAS las combinaciones en orden normal
for combo_name, variables in variable_combinations.items():
    print(f"\nProbando combinacion: {combo_name}")
    print(f"Variables: {variables}")
    
    try:
        X_combo = df_clean[variables].select_dtypes(include=[np.number])
        
        # Si no hay suficientes datos, saltar
        if len(X_combo) < 10:
            print("  No hay suficientes datos, saltando...")
            continue
            
        # Escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combo)
        
        print(f"  Datos preparados: {X_scaled.shape}")
        
        for n_clusters in range(2, 11):
            for metodo in ['kmeans', 'gaussian', 'hierarchical']:
                
                # Evaluar clustering con ambas metricas
                db_score, ch_score, labels = evaluar_clustering(X_scaled, n_clusters, metodo)
                
                if db_score is not None and ch_score is not None:
                    resultados.append({
                        'combinacion': combo_name,
                        'n_clusters': n_clusters,
                        'metodo': metodo,
                        'db_score': db_score,
                        'ch_score': ch_score,
                        'n_variables': len(variables),
                        'variables': str(variables)
                    })
                    
                    # Actualizar mejor configuracion segun Davies-Bouldin
                    if db_score < mejor_score_db_global:
                        mejor_score_db_global = db_score
                        mejor_configuracion_db = {
                            'combinacion': combo_name,
                            'n_clusters': n_clusters,
                            'metodo': metodo,
                            'db_score': db_score,
                            'ch_score': ch_score,
                            'variables': variables,
                            'labels': labels
                        }
                    
                    # Actualizar mejor configuracion segun Calinski-Harabasz
                    if ch_score > mejor_score_ch_global:
                        mejor_score_ch_global = ch_score
                        mejor_configuracion_ch = {
                            'combinacion': combo_name,
                            'n_clusters': n_clusters,
                            'metodo': metodo,
                            'db_score': db_score,
                            'ch_score': ch_score,
                            'variables': variables,
                            'labels': labels
                        }
                    
                    print(f"    {metodo:12} - {n_clusters} clusters: DB = {db_score:.4f} | CH = {ch_score:.1f}")
                    
    except Exception as e:
        print(f"Error con combinacion {combo_name}: {e}")
        continue

# Convertir resultados a DataFrame
df_resultados = pd.DataFrame(resultados)

# ====================================================================
# 5. ANALISIS DE RESULTADOS - COMPARATIVA DE METRICAS
# ====================================================================

print("\n" + "="*70)
print("RESULTADOS DEL EXPERIMENTO - COMPARATIVA DE METRICAS")
print("="*70)
print(f"Total de configuraciones probadas: {len(df_resultados)}")

if len(df_resultados) == 0:
    print("ERROR: No se pudieron calcular resultados. Verifica los datos.")
    exit()

# Top 10 mejores configuraciones segun cada metrica
print("\n TOP 10 MEJORES CONFIGURACIONES SEGUN DAVIES-BOULDIN (menor = mejor):")
top_10_db = df_resultados.nsmallest(10, 'db_score')
for i, (idx, row) in enumerate(top_10_db.iterrows(), 1):
    marca_especial = " *" if row['combinacion'] == 'combo7_config_especial' else ""
    print(f"{i:2d}. {row['metodo']:12} - {row['n_clusters']} clusters - "
          f"{row['combinacion']}{marca_especial} - DB: {row['db_score']:.4f} | CH: {row['ch_score']:.1f}")

print("\n TOP 10 MEJORES CONFIGURACIONES SEGUN CALINSKI-HARABASZ (mayor = mejor):")
top_10_ch = df_resultados.nlargest(10, 'ch_score')
for i, (idx, row) in enumerate(top_10_ch.iterrows(), 1):
    marca_especial = " *" if row['combinacion'] == 'combo7_config_especial' else ""
    print(f"{i:2d}. {row['metodo']:12} - {row['n_clusters']} clusters - "
          f"{row['combinacion']}{marca_especial} - DB: {row['db_score']:.4f} | CH: {row['ch_score']:.1f}")

# Mejores configuraciones globales
print("\n" + "="*70)
print("MEJORES CONFIGURACIONES GLOBALES:")
print("="*70)

print("\n  MEJOR CONFIGURACION SEGUN DAVIES-BOULDIN:")
print(f"   Metodo: {mejor_configuracion_db['metodo']}")
print(f"   Clusters: {mejor_configuracion_db['n_clusters']}")
print(f"   Combinacion: {mejor_configuracion_db['combinacion']}")
print(f"   Variables: {mejor_configuracion_db['variables']}")
print(f"   Davies-Bouldin: {mejor_configuracion_db['db_score']:.4f}")
print(f"   Calinski-Harabasz: {mejor_configuracion_db['ch_score']:.1f}")

print("\n  MEJOR CONFIGURACION SEGUN CALINSKI-HARABASZ:")
print(f"   Metodo: {mejor_configuracion_ch['metodo']}")
print(f"   Clusters: {mejor_configuracion_ch['n_clusters']}")
print(f"   Combinacion: {mejor_configuracion_ch['combinacion']}")
print(f"   Variables: {mejor_configuracion_ch['variables']}")
print(f"   Davies-Bouldin: {mejor_configuracion_ch['db_score']:.4f}")
print(f"   Calinski-Harabasz: {mejor_configuracion_ch['ch_score']:.1f}")

# Decidir cual usar (por defecto usamos la mejor segun Davies-Bouldin)
print("\n" + "="*70)
print("CONFIGURACION SELECCIONADA PARA APLICAR:")
print("="*70)
print("Se usara la mejor configuracion segun Davies-Bouldin")
mejor_configuracion = mejor_configuracion_db

# ====================================================================
# 6. VISUALIZACIONES MEJORADAS
# ====================================================================

print("\nGENERANDO VISUALIZACIONES...")

# Crear figura con mas subplots para ambas metricas
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('COMPARATIVA DE METRICAS DE CLUSTERING', fontsize=16, fontweight='bold')

# Subplot 1: Evolucion de Davies-Bouldin por metodo
ax1 = axes[0, 0]
for metodo in ['kmeans', 'gaussian', 'hierarchical']:
    metodo_data = df_resultados[df_resultados['metodo'] == metodo]
    if len(metodo_data) > 0:
        avg_scores = metodo_data.groupby('n_clusters')['db_score'].mean()
        ax1.plot(avg_scores.index, avg_scores.values, marker='o', label=metodo, linewidth=2)

ax1.set_xlabel('Numero de Clusters')
ax1.set_ylabel('Davies-Bouldin Score (menor = mejor)')
ax1.set_title('Evolucion de Davies-Bouldin por Metodo')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Evolucion de Calinski-Harabasz por metodo
ax2 = axes[0, 1]
for metodo in ['kmeans', 'gaussian', 'hierarchical']:
    metodo_data = df_resultados[df_resultados['metodo'] == metodo]
    if len(metodo_data) > 0:
        avg_scores = metodo_data.groupby('n_clusters')['ch_score'].mean()
        ax2.plot(avg_scores.index, avg_scores.values, marker='o', label=metodo, linewidth=2)

ax2.set_xlabel('Numero de Clusters')
ax2.set_ylabel('Calinski-Harabasz Score (mayor = mejor)')
ax2.set_title('Evolucion de Calinski-Harabasz por Metodo')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Mejores combinaciones de variables segun DB
ax3 = axes[1, 0]
combo_scores_db = df_resultados.groupby('combinacion')['db_score'].min().sort_values()
combo_scores_db.plot(kind='bar', color='lightcoral', ax=ax3)
ax3.set_title('Mejor Davies-Bouldin por Combinacion de Variables')
ax3.set_ylabel('Mejor Davies-Bouldin Score')
ax3.tick_params(axis='x', rotation=45)

# Subplot 4: Mejores combinaciones de variables segun CH
ax4 = axes[1, 1]
combo_scores_ch = df_resultados.groupby('combinacion')['ch_score'].max().sort_values(ascending=False)
combo_scores_ch.plot(kind='bar', color='lightblue', ax=ax4)
ax4.set_title('Mejor Calinski-Harabasz por Combinacion de Variables')
ax4.set_ylabel('Mejor Calinski-Harabasz Score')
ax4.tick_params(axis='x', rotation=45)

# Subplot 5: Comparacion de metodos para DB
ax5 = axes[2, 0]
metodos = df_resultados['metodo'].unique()
x_pos = np.arange(len(metodos))
width = 0.35

db_means = df_resultados.groupby('metodo')['db_score'].min().values
ch_means = df_resultados.groupby('metodo')['ch_score'].max().values

bars1 = ax5.bar(x_pos - width/2, db_means, width, label='DB (menor mejor)', color='lightcoral')
ax5.set_xlabel('Metodo')
ax5.set_ylabel('Davies-Bouldin Score')
ax5.set_title('Comparacion de Davies-Bouldin por Metodo')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(metodos)
ax5.legend()

# Subplot 6: Comparacion de metodos para CH
ax6 = axes[2, 1]
bars2 = ax6.bar(x_pos - width/2, ch_means, width, label='CH (mayor mejor)', color='lightblue')
ax6.set_xlabel('Metodo')
ax6.set_ylabel('Calinski-Harabasz Score')
ax6.set_title('Comparacion de Calinski-Harabasz por Metodo')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(metodos)
ax6.legend()

plt.tight_layout()
plt.show()

# ====================================================================
# 7. APLICAR MEJOR CONFIGURACION AL DATASET COMPLETO
# ====================================================================

print("\nAPLICANDO MEJOR CONFIGURACION AL DATASET...")

# Preparar datos con la mejor combinacion de variables
X_final = df_clean[mejor_configuracion['variables']].select_dtypes(include=[np.number])
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)

# Aplicar el mejor metodo
if mejor_configuracion['metodo'] == 'kmeans':
    model_final = KMeans(n_clusters=mejor_configuracion['n_clusters'], 
                        random_state=42, n_init=10)
elif mejor_configuracion['metodo'] == 'gaussian':
    model_final = GaussianMixture(n_components=mejor_configuracion['n_clusters'], 
                                 random_state=42)
else:  # hierarchical
    model_final = AgglomerativeClustering(n_clusters=mejor_configuracion['n_clusters'], 
                                         linkage='ward')

labels_final = model_final.fit_predict(X_final_scaled)

# Guardar resultados en el dataset
df_final = df_clean.copy()
df_final['mejor_cluster'] = labels_final

# Mostrar caracteristicas de los clusters
print("\nCARACTERISTICAS COMPLETAS DE LOS CLUSTERS:")
pd.set_option('display.max_columns', None)  # Mostrar TODAS las columnas
cluster_analysis = df_final.groupby('mejor_cluster')[mejor_configuracion['variables']].mean()
print(cluster_analysis)

# ====================================================================
# 8. GUARDAR RESULTADOS
# ====================================================================

# Guardar dataset con clusters
nombre_archivo = f"dataset_clusters_{mejor_configuracion['metodo']}_{mejor_configuracion['n_clusters']}clusters.csv"
df_final.to_csv(nombre_archivo, index=False)

# Guardar resultados del experimento
df_resultados.to_csv('resultados_experimento_completo.csv', index=False)

print(f"\n" + "="*70)
print("RESULTADOS GUARDADOS:")
print("="*70)
print(f"- Dataset con clusters: '{nombre_archivo}'")
print(f"- Resultados del experimento: 'resultados_experimento_completo.csv'")

print(f"\n" + "="*70)
print("EXPERIMENTO COMPLETADO!")
print("="*70)
print(f"   - Configuracion utilizada (segun Davies-Bouldin):")
print(f"   - Metodo: {mejor_configuracion['metodo']}")
print(f"   - Clusters: {mejor_configuracion['n_clusters']}") 
print(f"   - Combinacion: {mejor_configuracion['combinacion']}")
print(f"   - Variables: {mejor_configuracion['variables']}")
print(f"   - Davies-Bouldin: {mejor_configuracion['db_score']:.4f}")
print(f"   - Calinski-Harabasz: {mejor_configuracion['ch_score']:.1f}")