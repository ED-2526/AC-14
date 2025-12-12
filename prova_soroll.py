import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# =============================
# 1. CARREGAR DATASET
# =============================
df = pd.read_csv("treballadors_definitiu.csv")
print("Dataset carregat:", df.shape)

# Separar només columnes numèriques
df_num = df.select_dtypes(include=[np.number])
print("Columnes numèriques:", df_num.columns.tolist())
print(df_num.head())

# =============================
# 2. ANALITZAR VALORS PERDUTS
# =============================
missing = df.isna().mean().sort_values(ascending=False)
print("\nPercentatge de valors perduts per columna:")
print(missing)

plt.figure(figsize=(10,5))
missing.plot(kind='bar')
plt.title("Percentatge de NaN per variable")
plt.tight_layout()
plt.show()
"""

# =============================
# 3. FREQÜÈNCIA DEL VALOR MÉS COMÚ (NOMÉS NUMÈRIQUES)
# =============================
df_num = df.select_dtypes(include=[np.number])

freq_ratio = {}
for col in df_num.columns:
    most_freq = df_num[col].value_counts(normalize=True).iloc[0]
    freq_ratio[col] = most_freq

freq_ratio = pd.Series(freq_ratio).sort_values(ascending=False)
print("\nProporció del valor més freqüent per variable:")
print(freq_ratio)

plt.figure(figsize=(10,5))
freq_ratio.plot(kind='bar')
plt.title("Proporció del valor més freqüent per variable")
plt.tight_layout()
plt.show()
"""

# =============================
# 4. VARIÀNCIA DE CADA VARIABLE (NOMÉS NUMÈRIQUES)
# =============================
variances = df_num.var().sort_values()
print("\nVariància per variable (només numèriques):")
print(variances)

plt.figure(figsize=(10,5))
variances.plot(kind='bar')
plt.title("Variància de cada variable numèrica")
plt.tight_layout()
plt.show()

"""
# =============================
# 5. CORRELACIÓ ENTRE VARIABLES NUMÈRIQUES
# =============================
plt.figure(figsize=(14,10))
sns.heatmap(df_num.corr(), annot=False, cmap="coolwarm")
plt.title("Matriz de correlacions (variables numèriques)")
plt.tight_layout()
plt.show()

# =============================
# 6. PCA PER VEURE CONTRIBUCIÓ (NOMÉS NUMÈRIQUES)
# =============================
# (si hi ha NaN, els traiem només per al PCA)
df_num_pca = df_num.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num_pca)

pca = PCA()
pca.fit(X_scaled)

explained = pca.explained_variance_ratio_

plt.figure(figsize=(8,4))
plt.plot(np.cumsum(explained))
plt.xlabel("Número de components")
plt.ylabel("Variància explicada acumulada")
plt.title("PCA - Variància explicada acumulada")
plt.grid(True)
plt.tight_layout()
plt.show()
"""