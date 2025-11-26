import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# 1. Leer el dataset combinado
df_combinado = pd.read_csv('dataset_combinado.csv')

# 2. Filtrar solo los trabajadores de Estados Unidos (source = 2 y country = 'United States')
df_trabajadores_usa = df_combinado[(df_combinado['source'] == 2) & (df_combinado['country'] == 'United States')]

# 3. Preprocesar: Eliminar columnas no relevantes y manejar valores nulos
columnas_a_conservar = ['age', 'sex', 'self_employed', 'family_history', 'treatment', 'work_interfere', 
                        'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 
                        'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 
                        'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 
                        'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'state']

# Filtramos las columnas
df_trabajadores_usa = df_trabajadores_usa[columnas_a_conservar]

# 4. Separar las columnas numéricas y categóricas
columnas_numericas = df_trabajadores_usa.select_dtypes(include=['float64', 'int64']).columns
columnas_categoricas = df_trabajadores_usa.select_dtypes(include=['object']).columns

# 5. Imputación de valores nulos
# Imputar valores nulos en columnas numéricas con la media
imputer_numerico = SimpleImputer(strategy='mean')
df_trabajadores_usa[columnas_numericas] = imputer_numerico.fit_transform(df_trabajadores_usa[columnas_numericas])

# Imputar valores nulos en columnas categóricas con el valor más frecuente
imputer_categorico = SimpleImputer(strategy='most_frequent')
df_trabajadores_usa[columnas_categoricas] = imputer_categorico.fit_transform(df_trabajadores_usa[columnas_categoricas])

# 6. Escalado de las características numéricas
scaler = StandardScaler()
df_trabajadores_usa_scaled = pd.DataFrame(scaler.fit_transform(df_trabajadores_usa[columnas_numericas]), columns=columnas_numericas)

# 7. Aplicar el modelo de KMeans para hacer clustering binario (n_clusters=2)
kmeans = KMeans(n_clusters=2, random_state=42)
df_trabajadores_usa['cluster'] = kmeans.fit_predict(df_trabajadores_usa_scaled)

# 8. Reducir la dimensionalidad a 2D con PCA para la visualización
pca = PCA(n_components=2)
df_trabajadores_usa_pca = pca.fit_transform(df_trabajadores_usa_scaled)

# 9. Visualización de los clusters según el estado (coloreados por 'state')
plt.figure(figsize=(12, 8))

# Primer gráfico: visualización de los clusters (por 'cluster')
plt.subplot(1, 2, 1)
plt.scatter(df_trabajadores_usa_pca[:, 0], df_trabajadores_usa_pca[:, 1], 
            c=df_trabajadores_usa['cluster'], cmap='viridis')
plt.title('Clustering de Trabajadores de EE.UU. - Necesidad de Asistencia Psicológica (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')

# Segundo gráfico: visualización de los trabajadores según el estado (por 'state')
plt.subplot(1, 2, 2)
plt.scatter(df_trabajadores_usa_pca[:, 0], df_trabajadores_usa_pca[:, 1], 
            c=df_trabajadores_usa['state'].astype('category').cat.codes, cmap='tab20')
plt.title('Clustering de Trabajadores de EE.UU. - Según el Estado (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Estado')

plt.tight_layout()
plt.show()

# 10. Guardar el dataset con los clusters asignados en un archivo CSV
df_trabajadores_usa.to_csv('trabajadores_usa_clusters_binario.csv', index=False)

print("\n[EXITO] El dataset de trabajadores de EE.UU. con clustering binario ha sido guardado como 'trabajadores_usa_clusters_binario.csv'.")
