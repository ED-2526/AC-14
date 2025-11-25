import pandas as pd
import scipy.stats as stats
# Llegim el dataset
df_treballadors = pd.read_csv('treballadors.csv')

df_treballadors = df_treballadors.dropna(subset=['Gender', 'remote_work'])

# Crear una tabla de contingencia para las variables seleccionadas
contingency_table = pd.crosstab(df_treballadors['Gender'], df_treballadors['remote_work'])

# Realizar la prueba de Chi-cuadrado
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Mostrar los resultados
print("Chi2:", chi2)
print("P-valor:", p)
print("Grados de libertad:", dof)
print("Tabla esperada:", expected)

# Interpretar el resultado
if p < 0.05:
    print("Rechazamos la hipótesis nula: hay una relación significativa entre las variables.")
else:
    print("No se rechaza la hipótesis nula: no hay una relación significativa entre las variables.")