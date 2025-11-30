import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
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
# 3. FUNCION DE EVALUACION CON DAVIES-BOULDIN
# ====================================================================

def evaluar_clustering(X, n_clusters, metodo):
    """
    Evalua clustering usando Davies-Bouldin (menor = mejor)
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
            return None, None
            
        # Calcular Davies-Bouldin score
        db_score = davies_bouldin_score(X, labels)
        return db_score, labels
        
    except Exception as e:
        return None, None

# ====================================================================
# 4. EXPERIMENTO PRINCIPAL: BUSQUEDA DE LA MEJOR CONFIGURACION
# ====================================================================
# EXPERIMENTO PRINCIPAL: BUSQUEDA DE LA MEJOR CONFIGURACION
print("INICIANDO EXPERIMENTO DE CLUSTERING...")
print("Probando: 2-10 clusters x 3 metodos x 7 combinaciones de variables")

resultados = []
mejor_score_global = float('inf')
mejor_configuracion = None

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
                
                # Evaluar clustering
                db_score, labels = evaluar_clustering(X_scaled, n_clusters, metodo)
                
                if db_score is not None:
                    resultados.append({
                        'combinacion': combo_name,
                        'n_clusters': n_clusters,
                        'metodo': metodo,
                        'db_score': db_score,
                        'n_variables': len(variables),
                        'variables': str(variables)
                    })
                    
                    # Actualizar mejor configuracion global
                    if db_score < mejor_score_global:
                        mejor_score_global = db_score
                        mejor_configuracion = {
                            'combinacion': combo_name,
                            'n_clusters': n_clusters,
                            'metodo': metodo,
                            'db_score': db_score,
                            'variables': variables,
                            'labels': labels
                        }
                    
                    print(f"    {metodo:12} - {n_clusters} clusters: DB = {db_score:.4f}")
                    
    except Exception as e:
        print(f"Error con combinacion {combo_name}: {e}")
        continue

# Convertir resultados a DataFrame
df_resultados = pd.DataFrame(resultados)

# ====================================================================
# 5. ANALISIS DE RESULTADOS
# ====================================================================

print("RESULTADOS DEL EXPERIMENTO:")
print(f"Total de configuraciones probadas: {len(df_resultados)}")

if len(df_resultados) == 0:
    print("ERROR: No se pudieron calcular resultados. Verifica los datos.")
    exit()

# Resultados especificos de la configuracion especial
resultados_especial = df_resultados[df_resultados['combinacion'] == 'combo7_config_especial']
if len(resultados_especial) > 0:
    print(f"\nRESULTADOS CONFIGURACION ESPECIAL:")
    mejor_especial = resultados_especial.nsmallest(1, 'db_score').iloc[0]
    print(f"Mejor resultado: {mejor_especial['metodo']} - {mejor_especial['n_clusters']} clusters")
    print(f"Davies-Bouldin: {mejor_especial['db_score']:.4f}")

# Top 10 mejores configuraciones
top_10 = df_resultados.nsmallest(10, 'db_score')
print("\nTOP 10 MEJORES CONFIGURACIONES (Davies-Bouldin):")
for i, (idx, row) in enumerate(top_10.iterrows(), 1):
    marca_especial = " *" if row['combinacion'] == 'combo7_config_especial' else ""
    print(f"{i:2d}. {row['metodo']:12} - {row['n_clusters']} clusters - "
          f"{row['combinacion']}{marca_especial} - DB: {row['db_score']:.4f}")

# Mejor configuracion global
print("\nMEJOR CONFIGURACION GLOBAL:")
print(f"Metodo: {mejor_configuracion['metodo']}")
print(f"Clusters: {mejor_configuracion['n_clusters']}")
print(f"Combinacion: {mejor_configuracion['combinacion']}")
print(f"Variables: {mejor_configuracion['variables']}")
print(f"Score Davies-Bouldin: {mejor_configuracion['db_score']:.4f}")

# ====================================================================
# 6. VISUALIZACIONES
# ====================================================================

print("GENERANDO VISUALIZACIONES...")

plt.figure(figsize=(15, 10))

# Subplot 1: Mejor metodo por numero de clusters
plt.subplot(2, 2, 1)
for metodo in ['kmeans', 'gaussian', 'hierarchical']:
    metodo_data = df_resultados[df_resultados['metodo'] == metodo]
    if len(metodo_data) > 0:
        avg_scores = metodo_data.groupby('n_clusters')['db_score'].mean()
        plt.plot(avg_scores.index, avg_scores.values, marker='o', label=metodo, linewidth=2)

plt.xlabel('Numero de Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.title('Evolucion de Davies-Bouldin por Metodo')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Mejores combinaciones de variables
plt.subplot(2, 2, 2)
combo_scores = df_resultados.groupby('combinacion')['db_score'].min().sort_values()
combo_scores.plot(kind='bar', color='lightcoral')
plt.title('Mejor Score por Combinacion de Variables')
plt.ylabel('Mejor Davies-Bouldin Score')
plt.xticks(rotation=45)

# Subplot 3: Comparacion de metodos
plt.subplot(2, 2, 3)
metodo_scores = df_resultados.groupby('metodo')['db_score'].min()
metodo_scores.plot(kind='bar', color='lightgreen')
plt.title('Mejor Score por Metodo de Clustering')
plt.ylabel('Mejor Davies-Bouldin Score')

# Subplot 4: Resultados de configuracion especial
plt.subplot(2, 2, 4)
try:
    heatmap_data = df_resultados.pivot_table(values='db_score', 
                                           index='n_clusters', 
                                           columns='metodo', 
                                           aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, cmap='viridis_r', fmt='.3f',
                cbar_kws={'label': 'Davies-Bouldin Score'})
    plt.title('Heatmap: Clusters vs Metodos\n')
except:
    plt.text(0.5, 0.5, 'No se pudo generar heatmap', 
             ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# ====================================================================
# 7. APLICAR MEJOR CONFIGURACION AL DATASET COMPLETO
# ====================================================================

print("APLICANDO MEJOR CONFIGURACION AL DATASET...")

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
print("CARACTERISTICAS COMPLETAS DE LOS CLUSTERS:")
pd.set_option('display.max_columns', None)  # Mostrar TODAS las columnas
cluster_analysis = df_final.groupby('mejor_cluster')[mejor_configuracion['variables']].mean()
print(cluster_analysis)

# ====================================================================
# 8. GUARDAR RESULTADOS
# ====================================================================

# Guardar dataset con clusters
#df_final.to_csv('dataset_con_mejor_clustering.csv', index=False)
df_resultados.to_csv('resultados_experimento_clustering.csv', index=False)

print("RESULTADOS GUARDADOS:")
print("- Dataset con clusters: 'dataset_con_mejor_clustering.csv'")
print("- Resultados del experimento: 'resultados_experimento_clustering.csv'")

print("EXPERIMENTO COMPLETADO!")
print("Mejor configuracion encontrada:")
print(f"   - Metodo: {mejor_configuracion['metodo']}")
print(f"   - Clusters: {mejor_configuracion['n_clusters']}") 
print(f"   - Combinacion: {mejor_configuracion['combinacion']}")
print(f"   - Variables: {mejor_configuracion['variables']}")
print(f"   - Davies-Bouldin: {mejor_configuracion['db_score']:.4f}")

