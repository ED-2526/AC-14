import pandas as pd

# Llegir els datasets
df_estudiants = pd.read_csv('estudiants_med.csv')
df_treballadors = pd.read_csv('treballadors.csv')

# 1. REINICIAR IDs EN STUDENTS
df_estudiants['id'] = range(1, len(df_estudiants) + 1)

# 2. CALCULAR Y ASIGNAR IDs PARA WORKERS
next_id = len(df_estudiants) + 1
# Afegir una columna 'source' per identificar d'on provenen les dades (1 per estudiants, 2 per treballadors)
df_estudiants['source'] = 1  # Els estudiants de medicina seran 1
df_treballadors['source'] = 2  # Els treballadors seran 2

# Eliminar les files amb valors nuls en 'Age' i 'Gender' abans d'assignar els IDs als treballadors
df_treballadors = df_treballadors.dropna(subset=['Age', 'Gender'])

# Assignar un ID únic als treballadors
next_id = df_estudiants['id'].max() + 1  # L'ID següent serà l'últim ID dels estudiants + 1
df_treballadors['id'] = range(next_id, next_id + len(df_treballadors))

# 3. AFEGIR COLUMNA 'source'
df_estudiants['source'] = 1  # Estudiantes
df_treballadors['source'] = 2  # Trabajadores

# 4. FUNCIÓN PARA CODIFICAR GÉNERO DE WORKERS
def codificar_sexo(valor):
    if pd.isna(valor):
        return 3  # Otro para valores nulos
    
    valor = str(valor).lower().strip()
    
    # Male (1)
    male_keywords = ['male', 'm', 'Male', 'man', 'Cis Male', 'cis man', 'Male-ish', 'mail', 'mal', 'maile', 'make', 'M']
    # Female (2)  
    female_keywords = ['female', 'f', 'woman', 'Cis-Female',  'femake', 'femail']
    # Trans/Non-binary (3)
    trans_keywords = ['trans', 'non-binary', 'nb', 'genderqueer', 'androgynous', 'agender', 'fluid', 'queer']
    
    if any(keyword in valor for keyword in male_keywords):
        return 1
    elif any(keyword in valor for keyword in female_keywords):
        return 2
    elif any(keyword in valor for keyword in trans_keywords):
        return 3
    else:
        return 3  # Otro para cualquier otro caso
    
def codificar_self_employed(valor):
    if pd.isna(valor):
        return int(2)  # Otro para valores nulos
    
    valor = str(valor).lower().strip()
    
    if valor in ['yes', 'y', 'true', '1']:
        return int(1)
    elif valor in ['no', 'n', 'false', '0']:
        return int(0)
    else:
        return int(2)  # Otro para cualquier otro caso

def codificar_family_history(valor):
    if pd.isna(valor):
        return int(2)  # Otro para valores nulos
    
    valor = str(valor).lower().strip()
    
    if valor in ['yes', 'y', 'true', '1']:
        return int(1)
    elif valor in ['no', 'n', 'false', '0']:
        return int(0)
    else:
        return int(2)  # Otro para cualquier otro caso
    
def codificar_treatment(valor):
    if pd.isna(valor):
        return int(2)  # Otro para valores nulos
    
    valor = str(valor).lower().strip()
    
    if valor in ['yes', 'y', 'true', '1']:
        return int(1)
    elif valor in ['no', 'n', 'false', '0']:
        return int(0)
    else:
        return int(2)  # Otro para cualquier otro caso
    
def codificar_work_interfere(valor):
    mapping = {
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Often': 3
    }
    return mapping.get(valor, 4)  # 4 para otros casos o nulos

def codificar_no_employees(valor):
    mapping = {
        '1-5': 0,
        '6-25': 1,
        '26-100': 2,
        '100-500': 3,
        '500-1000': 4,
        '1000-5000': 5,
        '5000-10000': 6,
        '10000+': 7,
        'More than 1000': 5
    }
    return mapping.get(valor, 8)  # 8 para otros casos o nulos

def codificar_si_no_otro(valor):
    if pd.isna(valor):
        return int(2)  # Otro para valores nulos
    
    valor = str(valor).lower().strip()
    
    if valor in ['yes', 'y', 'true', '1']:
        return int(1)
    elif valor in ['no', 'n', 'false', '0']:
        return int(0)
    else:
        return int(2)  # Otro para cualquier otro caso
    
def codificar_leave(valor):
    mapping = {
        'Very easy': 0,
        'Somewhat easy': 1,
        'Somewhat difficult': 2,
        'Very difficult': 3
    }
    return mapping.get(valor, 4)  # 4 para otros casos o nulos


# 5. APLICAR CODIFICACIÓN A VALORES DE WORKERS Y MODIFICAR COLUMNAS'
df_treballadors['sex'] = df_treballadors['Gender'].apply(codificar_sexo).astype(int)

df_treballadors['sex'] = df_treballadors['sex'].apply(codificar_sexo).astype(int)

df_treballadors['self_employed'] = df_treballadors['self_employed'].apply(codificar_self_employed).astype(int)

df_treballadors['family_history'] = df_treballadors['family_history'].apply(codificar_family_history).astype(int)

df_treballadors['treatment'] = df_treballadors['treatment'].apply(codificar_treatment).astype(int)

df_treballadors['work_interfere'] = df_treballadors['work_interfere'].apply(codificar_work_interfere).astype(int)

df_treballadors['no_employees'] = df_treballadors['no_employees'].apply(codificar_no_employees).astype(int)

df_treballadors['remote_work'] = df_treballadors['remote_work'].apply(codificar_si_no_otro).astype(int)
df_treballadors['tech_company'] = df_treballadors['tech_company'].apply(codificar_si_no_otro).astype(int)
df_treballadors['benefits'] = df_treballadors['benefits'].apply(codificar_si_no_otro).astype(int)
df_treballadors['care_options'] = df_treballadors['care_options'].apply(codificar_si_no_otro).astype(int)
df_treballadors['wellness_program'] = df_treballadors['wellness_program'].apply(codificar_si_no_otro).astype(int)
df_treballadors['seek_help'] = df_treballadors['seek_help'].apply(codificar_si_no_otro).astype(int)
df_treballadors['anonymity'] = df_treballadors['anonymity'].apply(codificar_si_no_otro).astype(int)
df_treballadors['leave'] = df_treballadors['leave'].apply(codificar_leave).astype(int)
df_treballadors['mental_health_consequence'] = df_treballadors['mental_health_consequence'].apply(codificar_si_no_otro).astype(int)
df_treballadors['phys_health_consequence'] = df_treballadors['phys_health_consequence'].apply(codificar_si_no_otro).astype(int)
df_treballadors['coworkers'] = df_treballadors['coworkers'].apply(codificar_si_no_otro).astype(int)
df_treballadors['supervisor'] = df_treballadors['supervisor'].apply(codificar_si_no_otro).astype(int)
df_treballadors['mental_health_interview'] = df_treballadors['mental_health_interview'].apply(codificar_si_no_otro).astype(int)
df_treballadors['phys_health_interview'] = df_treballadors['phys_health_interview'].apply(codificar_si_no_otro).astype(int)
df_treballadors['mental_vs_physical'] = df_treballadors['mental_vs_physical'].apply(codificar_si_no_otro).astype(int)
df_treballadors['obs_consequence'] = df_treballadors['obs_consequence'].apply(codificar_si_no_otro).astype(int)






# 6. RENOMBRAR COLUMNAS DE WORKERS (eliminamos Gender ya que tenemos sex)
df_treballadors_renamed = df_treballadors.rename(columns={
    'Age': 'age',
    'Country': 'country',
    'sex': 'sex',
    'state': 'state',
    'self_employed': 'self_employed',
    'family_history': 'family_history',
    'treatment': 'treatment',
    'work_interfere': 'work_interfere',
    'no_employees': 'no_employees',
    'remote_work': 'remote_work',
    'tech_company': 'tech_company',
    'benefits': 'benefits',
    'care_options': 'care_options',
    'wellness_program': 'wellness_program',
    'seek_help': 'seek_help',
    'anonymity': 'anonymity',
    'leave': 'leave',
    'mental_health_consequence': 'mental_health_consequence',
    'phys_health_consequence': 'phys_health_consequence',
    'coworkers': 'coworkers',
    'supervisor': 'supervisor',
    'mental_health_interview': 'mental_health_interview',
    'phys_health_interview': 'phys_health_interview',
    'mental_vs_physical': 'mental_vs_physical',
    'obs_consequence': 'obs_consequence',
    'comments': 'comments'
})

# Eliminar la columna Gender original ya que tenemos sex
df_treballadors_renamed = df_treballadors_renamed.drop('Gender', axis=1)

# 7. REORDENAR COLUMNAS - ID y SOURCE primero
columnas_base = ['id', 'source', 'age', 'sex']
columnas_restantes_est = [col for col in df_estudiants.columns if col not in columnas_base]
columnas_estudiantes = columnas_base + columnas_restantes_est
df_estudiants = df_estudiants[columnas_estudiantes]

# Para workers: id, source, age, sex primero, luego el resto (no tienen year)
columnas_restantes_work = [col for col in df_treballadors_renamed.columns if col not in columnas_base]
columnas_trabajadores = columnas_base + columnas_restantes_work
df_treballadors_renamed = df_treballadors_renamed[columnas_trabajadores]

# 8. COMBINAR MANTENIENDO TODAS LAS COLUMNAS
df_combinat = pd.concat([df_estudiants, df_treballadors_renamed], ignore_index=True, sort=False)

# 9. VERIFICACIÓN
print("=== VERIFICACION DE IDs Y SEX ===")
print(f"Students: IDs {df_estudiants['id'].min()} a {df_estudiants['id'].max()}")
print(f"Workers: IDs {df_treballadors_renamed['id'].min()} a {df_treballadors_renamed['id'].max()}")
print(f"Combinado: IDs {df_combinat['id'].min()} a {df_combinat['id'].max()}")

print("\n=== DISTRIBUCION DE SEX EN EL DATASET COMBINADO ===")
distribucion_sex = df_combinat['sex'].value_counts().sort_index()
print(distribucion_sex)
print("\nCodificacion:")
print("1 = Male, 2 = Female, 3 = Otro/Trans/Non-binary")

print("\n=== DISTRIBUCION POR FUENTE ===")
print("Students - sex distribution:")
print(df_estudiants['sex'].value_counts().sort_index())
print("\nWorkers - sex distribution:")
print(df_treballadors_renamed['sex'].value_counts().sort_index())

print("\n=== MUESTRA DEL DATASET COMBINADO (primeras columnas) ===")
print(df_combinat[['id', 'source', 'sex', 'age']].head(10))

# 10. VERIFICAR INTEGRIDAD
print(f"\n=== INTEGRIDAD ===")
print(f"IDs unicos: {df_combinat['id'].nunique()}")
print(f"Total registros: {len(df_combinat)}")
print(f"IDs unicos?: {df_combinat['id'].nunique() == len(df_combinat)}")
print(f"Valores unicos en sex: {sorted(df_combinat['sex'].unique())}")
print(f"Valores nulos en sex: {df_combinat['sex'].isnull().sum()}")
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

# 11. GUARDAR EL DATASET COMBINADO
df_combinat.to_csv('dataset_combinado.csv', index=False)
print("\n[EXITO] Dataset combinado guardado como 'dataset_combinado.csv'")