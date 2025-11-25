import pandas as pd

# Llegir els datasets
df_estudiants = pd.read_csv('estudiants_med.csv')
df_treballadors = pd.read_csv('treballadors.csv')

# 1. REINICIAR IDs EN STUDENTS
df_estudiants['id'] = range(1, len(df_estudiants) + 1)

# 2. CALCULAR Y ASIGNAR IDs PARA WORKERS
next_id = len(df_estudiants) + 1
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

# 5. APLICAR CODIFICACIÓN AL GÉNERO DE WORKERS Y CREAR COLUMNA 'sex'
df_treballadors['sex'] = df_treballadors['Gender'].apply(codificar_sexo)

# 6. RENOMBRAR COLUMNAS DE WORKERS (eliminamos Gender ya que tenemos sex)
df_treballadors_renamed = df_treballadors.rename(columns={
    'Age': 'age',
    'Country': 'country',
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

# 11. GUARDAR EL DATASET COMBINADO
df_combinat.to_csv('dataset_combinado.csv', index=False)
print("\n[EXITO] Dataset combinado guardado como 'dataset_combinado.csv'")


