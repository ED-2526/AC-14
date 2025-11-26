import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Cargar el dataset combinado
df_combinat = pd.read_csv('dataset_combinado.csv')

# Eliminar la columna Timestamp que contiene fechas
df_combinat = df_combinat.drop(columns=['Timestamp'])

# Eliminar las columnas no necesarias para el clustering
df_clustering = df_combinat.drop(columns=['id', 'source', 'comments'])

# Eliminar solo las filas con valores nulos en las columnas críticas (por ejemplo, 'age' y 'sex')
df_clustering = df_clustering.dropna(subset=['age', 'sex'])

# Codificar la columna 'Country' usando LabelEncoder
label_encoder = LabelEncoder()
df_combinat['country'] = label_encoder.fit_transform(df_combinat['country'])

# Eliminar la columna 'state' y 'comments' si no son necesarias
df_clustering = df_combinat.drop(columns=['state', 'comments'])

# Verificar si el dataframe tiene filas
print(f"El dataframe tiene {df_clustering.shape[0]} filas y {df_clustering.shape[1]} columnas.")

# Verificar si hay valores NaN en las columnas
print(df_clustering.isnull().sum())

# Si hay valores NaN, se rellenan con la media (puedes usar la mediana si prefieres)
imputer = SimpleImputer(strategy='mean')
df_clustering = pd.DataFrame(imputer.fit_transform(df_clustering), columns=df_clustering.columns)

# Verificar si los NaN han sido eliminados
print(f"Valores nulos después de imputar: {df_clustering.isnull().sum()}")

# Escalar los datos (escalado de las variables para que todas tengan la misma importancia)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)

# 1. Determinar el número óptimo de clusters usando el método del codo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Visualizar el gráfico del codo
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo para Determinar el Número de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.show()

# A partir del gráfico, seleccionamos el número de clusters adecuado (ej. 3 clusters)

# 2. Aplicar K-means con el número de clusters seleccionado
kmeans = KMeans(n_clusters=3, random_state=42)
df_combinat['cluster'] = kmeans.fit_predict(df_scaled)

# 3. Analizar los resultados
# Mostrar el número de muestras en cada cluster
print(df_combinat['cluster'].value_counts())

# Visualizar los clusters usando dos de las características (por ejemplo, 'age' y 'sex')
plt.figure(figsize=(10,6))
sns.scatterplot(x=df_combinat['age'], y=df_combinat['sex'], hue=df_combinat['cluster'], palette='Set2', s=100, alpha=0.7)
plt.title('Distribución de Clusters por Edad y Sexo')
plt.xlabel('Edad')
plt.ylabel('Sexo')
plt.legend(title='Cluster')
plt.show()

# 4. Ver las características promedio de cada cluster
cluster_means = df_combinat.groupby('cluster').mean()
print(cluster_means)
