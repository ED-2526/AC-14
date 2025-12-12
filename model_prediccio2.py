import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset combinado
df_combinat = pd.read_csv('dataset_combinado.csv')

# Eliminar solo las columnas necesarias una vez
columns_to_drop = ['id', 'source', 'comments', 'Timestamp']
df_combinat = df_combinat.drop(columns=[col for col in columns_to_drop if col in df_combinat.columns])

# Eliminar las columnas no necesarias para el clustering
df_clustering = df_combinat.drop(columns=['id', 'source', 'comments'])

# Rellenar los valores NaN para columnas numéricas
df_clustering_numeric = df_clustering.select_dtypes(include=['float64', 'int64'])
imputer_numeric = SimpleImputer(strategy='mean')
df_clustering[df_clustering_numeric.columns] = imputer_numeric.fit_transform(df_clustering[df_clustering_numeric.columns])

# Rellenar los valores NaN para columnas categóricas con la moda (valor más frecuente)
cat_columns = df_clustering.select_dtypes(include=['object']).columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
df_clustering[cat_columns] = imputer_categorical.fit_transform(df_clustering[cat_columns])

# Convertir las columnas categóricas a variables dummy usando OneHotEncoder
encoder = OneHotEncoder(drop='first')  # Sin sparse=False
df_encoded = encoder.fit_transform(df_clustering[cat_columns]).toarray()  # Convertir la matriz dispersa a densa

# Obtener nombres de las columnas codificadas
df_encoded_columns = encoder.get_feature_names_out(cat_columns)
df_encoded_df = pd.DataFrame(df_encoded, columns=df_encoded_columns)

# Unir las columnas codificadas con las numéricas
df_clustering_encoded = pd.concat([df_clustering[df_clustering_numeric.columns], df_encoded_df], axis=1)

# Escalar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering_encoded)

# K-means con k=4 (el número de clusters elegido)
kmeans = KMeans(n_clusters=4, random_state=42)
df_combinat['cluster'] = kmeans.fit_predict(df_scaled)

# Mostrar el número de muestras en cada cluster
print("Número de muestras por cluster:")
print(df_combinat['cluster'].value_counts())

# Analizar las características promedio de cada cluster
cluster_means = df_combinat.groupby('cluster').mean()
print("\nPromedio de características por cluster:")
print(cluster_means)

# Visualizar los clusters usando dos características (por ejemplo, 'age' y 'sex')
plt.figure(figsize=(10,6))
sns.scatterplot(x=df_combinat['age'], y=df_combinat['sex'], hue=df_combinat['cluster'], palette='Set2', s=100, alpha=0.7)
plt.title('Distribución de Clusters por Edad y Sexo')
plt.xlabel('Edad')
plt.ylabel('Sexo')
plt.legend(title='Cluster')
plt.show()
