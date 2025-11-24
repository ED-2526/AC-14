import pandas as pd

# Llegir els datasets
df_estudiants = pd.read_csv('estudiants_med.csv')
df_treballadors = pd.read_csv('treballadors.csv')

# Afegir una columna 'source' per identificar d'on provenen les dades (1 per estudiants, 2 per treballadors)
df_estudiants['source'] = 1  # Els estudiants de medicina seran 1
df_treballadors['source'] = 2  # Els treballadors seran 2

# Assignar un ID únic als treballadors (els estudiants ja tenen ID)
# Generem l'ID a partir de l'últim ID dels estudiants + 1
next_id = df_estudiants['id'].max() + 1  # L'ID següent serà l'últim ID dels estudiants + 1

# Afegir ID als treballadors, assegurant-nos que no hi hagi NaN
df_treballadors['id'] = range(next_id, next_id + len(df_treballadors))

# Comprovar si els IDs s'han afegit correctament
print(df_treballadors[['id']].head())

# Opcional: Seleccionar les columnes d'interès dels treballadors per igualar-les amb les dels estudiants
df_treballadors = df_treballadors[['Age', 'Gender', 'self_employed', 'family_history', 'treatment', 'work_interfere', 
                                   'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 
                                   'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 
                                   'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 
                                   'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'comments']]

# Renombrar les columnes dels treballadors perquè coincideixin amb les dels estudiants
df_treballadors.columns = ['age', 'sex', 'self_employed', 'family_history', 'treatment', 'work_interfere', 
                           'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 
                           'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 
                           'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 
                           'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'comments']

# Combinar els dos datasets
df_combinat = pd.concat([df_estudiants, df_treballadors], ignore_index=True)

# Comprovar els primers resultats
print(df_combinat.tail())   
