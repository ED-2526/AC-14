import pandas as pd
import scipy.stats as stats

# Definir las columnas que quieres comparar
variable_a_comparar_1 = 'Gender'      
variable_a_comparar_2 = 'remote_work'  # Variable 2 (por ejemplo, 'remote_work')

# Llegim el dataset
df_treballadors = pd.read_csv('treballadors.csv')

# Eliminar las filas con valores nulos en las columnas que vamos a analizar
df_treballadors = df_treballadors.dropna(subset=[variable_a_comparar_1, variable_a_comparar_2])

# Crear una tabla de contingencia para las dos variables seleccionadas
contingency_table = pd.crosstab(df_treballadors[variable_a_comparar_1], df_treballadors[variable_a_comparar_2])

# Realizar la prueba de Chi-cuadrado
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Mostrar los resultados
print(f"Comparant les variables: '{variable_a_comparar_1}' i '{variable_a_comparar_2}'")
print("Chi2:", chi2)
print("P-valor:", p)
print("Graus de llibertat:", dof)
print("Taula esperada:", expected)

# Interpretar el resultado
if p < 0.05:
    print("Rebutgem la hipòtesi nul·la: hi ha una relació significativa entre les variables.")
else:
    print("No rebutgem la hipòtesi nul·la: no hi ha una relació significativa entre les variables.")
