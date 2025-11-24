import pandas as pd

# Llegir els datasets
df_estudiants = pd.read_csv('estudiants_med.csv')
df_treballadors = pd.read_csv('treballadors.csv')

# Afegir la columna 'source' per identificar el tipus (1 per estudiants, 2 per treballadors)
df_estudiants['source'] = 1  # Els estudiants de medicina seran 1
df_treballadors['source'] = 2  # Els treballadors seran 2

# Afegir un identificador únic als treballadors
# Generem un identificador únic per als treballadors que no coincideixi amb els dels estudiants
next_id = df_estudiants['id'].max() + 1  # El següent id serà l'últim id d'estudiants + 1

# Assignem els ids als treballadors
df_treballadors['id'] = range(next_id, next_id + len(df_treballadors))

# Combinar els dos datasets
df_combinat = pd.concat([df_estudiants, df_treballadors], ignore_index=True)

# Veure les primeres files del dataframe combinat
print(df_combinat.head())