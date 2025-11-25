import pandas as pd

# Llegir els datasets
df_estudiants = pd.read_csv('estudiants_med.csv')
df_treballadors = pd.read_csv('treballadors.csv')

# Afegir una columna 'source' per identificar d'on provenen les dades (1 per estudiants, 2 per treballadors)
df_estudiants['source'] = 1  # Els estudiants de medicina seran 1
df_treballadors['source'] = 2  # Els treballadors seran 2

# Eliminar les files amb valors nuls en 'Age' i 'Gender' abans d'assignar els IDs als treballadors
df_treballadors = df_treballadors.dropna(subset=['Age', 'Gender'])

# Assignar un ID únic als treballadors
next_id = df_estudiants['id'].max() + 1  # L'ID següent serà l'últim ID dels estudiants + 1
df_treballadors['id'] = range(next_id, next_id + len(df_treballadors))

# Comprovar que els IDs s'han assignat correctament (eliminant NaN)
print(df_treballadors[['id']].head())

# Convertir la columna 'id' a enters, per evitar qualsevol valor no numèric
df_treballadors['id'] = df_treballadors['id'].astype(int)

# Eliminar qualsevol fila amb NaN a la columna 'id' (per si es va filtrar alguna cosa incorrecta)
df_treballadors = df_treballadors.dropna(subset=['id'])

# Seleccionar només les columnes d'interès dels treballadors per igualar-les amb les dels estudiants
df_treballadors = df_treballadors[['source','id','Age', 'Gender', 'self_employed', 'family_history', 'treatment', 'work_interfere', 
                                   'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 
                                   'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 
                                   'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 
                                   'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'comments']]

# Renombrar les columnes dels treballadors perquè coincideixin amb les dels estudiants
df_treballadors.columns = ['source','id','age', 'sex', 'self_employed', 'family_history', 'treatment', 'work_interfere', 
                           'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 
                           'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 
                           'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 
                           'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'comments']

# Combinar els dos datasets
df_combinat = pd.concat([df_estudiants, df_treballadors], ignore_index=True)

# Reorganitzar les columnes per posar 'source' al principi
cols = ['source'] + [col for col in df_combinat.columns if col != 'source']
df_combinat = df_combinat[cols]

# Comprovar els primers resultats
print(df_combinat)

# Opcional: Guardar el resultat final en un nou arxiu CSV
df_combinat.to_csv('df_combinat.csv', index=False)