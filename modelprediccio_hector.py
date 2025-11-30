import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Cargar el dataset combinado
df_dataset = pd.read_csv('treballadors_definitiu.csv')

# Verificar las columnas disponibles
print(f"\nEl dataset tiene {df_dataset.shape[0]} filas y {df_dataset.shape[1]} columnas.")

# Eliminar solo las filas con valores nulos en las columnas criticas
df_clustering = df_dataset.dropna(subset=['Age', 'Gender'])

# VERIFICAR NULOS ANTES DE LLENARLOS
print("\n=== NUMEROS DE VALORES NULOS ANTES DE PROCESAR ===")
print("Valores nulos en Country:", df_clustering['Country'].isnull().sum())
print("Valores nulos en state:", df_clustering['state'].isnull().sum())

# Codificar las columnas categoricas (Country y state) usando LabelEncoder
label_encoder_country = LabelEncoder()
label_encoder_state = LabelEncoder()

# Si hay valores nulos en Country o state, los llenamos con 'Unknown' antes de codificar
df_clustering['Country'] = df_clustering['Country'].fillna('Unknown')
df_clustering['state'] = df_clustering['state'].fillna('Unknown')

# Aplicar LabelEncoder
df_clustering['Country_encoded'] = label_encoder_country.fit_transform(df_clustering['Country'])
df_clustering['state_encoded'] = label_encoder_state.fit_transform(df_clustering['state'])

# Seleccionar solo las columnas numericas para el clustering
columnas_numericas = ['Age', 'Gender', 'self_employed', 'family_history', 'treatment', 
                     'work_interfere', 'no_employees', 'remote_work', 'tech_company', 
                     'benefits', 'care_options', 'wellness_program', 'seek_help', 
                     'anonymity', 'leave', 'mental_health_consequence', 
                     'phys_health_consequence', 'coworkers', 'supervisor', 
                     'mental_health_interview', 'phys_health_interview', 
                     'mental_vs_physical', 'obs_consequence', 'Country_encoded', 'state_encoded']

df_numerico = df_clustering[columnas_numericas]

# Si hay valores NaN, se rellenan con la media
imputer = SimpleImputer(strategy='mean')
df_numerico_imputed = pd.DataFrame(imputer.fit_transform(df_numerico), 
                                  columns=df_numerico.columns)

# Verificar si los NaN han sido eliminados
print(f"\nValores nulos despues de imputar: {df_numerico_imputed.isnull().sum().sum()}")

# Escalar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numerico_imputed)

print(f"Datos escalados: {df_scaled.shape}")

# METODO DEL CODO - SOLO PARA DETERMINAR NUMERO OPTIMO DE CLUSTERS
print("\n=== METODO DEL CODO ===")
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    print(f"Clusters: {k} - Inercia: {kmeans.inertia_:.2f}")

# Visualizar el grafico del codo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--', linewidth=2, markersize=8)
plt.title('Metodo del Codo para Determinar el Numero Optimo de Clusters', fontsize=14)
plt.xlabel('Numero de Clusters', fontsize=12)
plt.ylabel('Inercia (Within-Cluster Sum of Squares)', fontsize=12)
plt.grid(True, alpha=0.3)

# Anadir valores en los puntos
for i, v in enumerate(inertia):
    plt.text(i + 1, v + 50, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Analisis automatico del punto de codo
print("\n=== ANALISIS DEL PUNTO DE CODO ===")
diferencias = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
diferencia_de_diferencias = [diferencias[i] - diferencias[i+1] for i in range(len(diferencias)-1)]

print("Diferencias de inercia entre clusters consecutivos:")
for i, diff in enumerate(diferencias):
    print(f"  De {i+1} a {i+2} clusters: {diff:.2f}")

print("\nDiferencias de las diferencias (aceleracion):")
for i, diff_diff in enumerate(diferencia_de_diferencias):
    print(f"  Cambio en {i+2} clusters: {diff_diff:.2f}")

# El punto de codo suele ser donde la segunda diferencia es maxima
if diferencia_de_diferencias:
    codo_recomendado = diferencia_de_diferencias.index(max(diferencia_de_diferencias)) + 2
    print(f"\nNUMERO DE CLUSTERS RECOMENDADO por analisis automatico: {codo_recomendado}")
else:
    print("\nNo se pudo calcular automaticamente el punto de codo")

# Interpretacion del grafico
print("\n=== INTERPRETACION ===")
print("Busca el punto donde la linea deja de descender abruptamente y comienza a aplanarse.")
print("Ese 'codo' indica el numero optimo de clusters.")
print("Valores tipicos suelen estar entre 2 y 5 clusters para datos de salud mental.")

print(f"\nRESUMEN FINAL:")
print(f"- Inercia minima: {min(inertia):.2f}")
print(f"- Inercia maxima: {max(inertia):.2f}")