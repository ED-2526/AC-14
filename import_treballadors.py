import pandas as pd
import numpy as np # Importació implícita per a pd.isna()


df_treballadors = pd.read_csv('treballadors.csv')
print("Dataset de Treballadors carregat.")


# ====================================================================
# 2. DEFINICIÓ DE LES FUNCIONS DE CODIFICACIÓ
# ====================================================================
df_treballadors = df_treballadors[df_treballadors['Age'] < 100].copy()

# Eliminar columnes innecessàries
df_treballadors = df_treballadors.drop(['Timestamp', 'comments'], axis=1)
print("Columnes 'Timestamp' i 'comments' eliminades.")


# Funció per codificar Gender. Mapeig desitjat:
# 1: Mascle (Male)
# 2: Dona (Female)
# 3: No binari / Altre / Nul
def codificar_sexo(valor):
    if pd.isna(valor):
        return 3
    
    valor = str(valor).lower().strip()
    
    # Male (1)
    # S'han netejat les llistes per evitar col·lisions
    male_keywords = ['male', 'm', 'man', 'cis male', 'cis man', 'male-ish', 'mail', 'mal', 'maile', 'make']
    # Female (2)  
    female_keywords = ['female', 'f', 'woman', 'cis-female', 'femake', 'femail']
    # Trans/Non-binary (3)
    trans_keywords = ['trans', 'non-binary', 'nb', 'genderqueer', 'androgynous', 'agender', 'fluid', 'queer', 'other']
    
    # MODIFICACIÓ CLAU: Prioritzem la cerca de 'female' i 'trans' per evitar que 'male' sigui una subcadena
    # d'algunes definicions més llargues que puguin contenir 'm'.
    
    if any(keyword in valor for keyword in female_keywords):
        return 2
    elif any(keyword in valor for keyword in male_keywords):
        return 1
    elif any(keyword in valor for keyword in trans_keywords):
        return 3
    else:
        return 3
    


# Funció per codificar Sí/No/Altre -> (1=Yes, 0=No, 2=Altre/Nul)
# S'han mantingut les funcions de l'usuari tot i la redundància
def codificar_self_employed(valor):
    if pd.isna(valor): return int(2)
    valor = str(valor).lower().strip()
    if valor in ['yes', 'y', 'true', '1']: return int(1)
    elif valor in ['no', 'n', 'false', '0']: return int(0)
    else: return int(2)

def codificar_family_history(valor):
    if pd.isna(valor): return int(2)
    valor = str(valor).lower().strip()
    if valor in ['yes', 'y', 'true', '1']: return int(1)
    elif valor in ['no', 'n', 'false', '0']: return int(0)
    else: return int(2)

def codificar_treatment(valor):
    if pd.isna(valor): return int(2)
    valor = str(valor).lower().strip()
    if valor in ['yes', 'y', 'true', '1']: return int(1)
    elif valor in ['no', 'n', 'false', '0']: return int(0)
    else: return int(2)

# Funció per codificar Work_interfere (Ordinal)
def codificar_work_interfere(valor):
    mapping = {
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Often': 3
    }
    return mapping.get(valor, 4)

# Funció per codificar No_employees (Ordinal)
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
    return mapping.get(valor, 8)

# Funció genèrica per codificar Sí/No/Altre (utilitzada per la majoria de binàries)
def codificar_si_no_otro(valor):
    if pd.isna(valor): return int(2)
    valor = str(valor).lower().strip()
    if valor in ['yes', 'y', 'true', '1']: return int(1)
    elif valor in ['no', 'n', 'false', '0']: return int(0)
    else: return int(2)

# Funció per codificar Leave (Ordinal)
def codificar_leave(valor):
    mapping = {
        'Very easy': 0,
        'Somewhat easy': 1,
        'Somewhat difficult': 2,
        'Very difficult': 3
    }
    return mapping.get(valor, 4)

# ====================================================================
# 3. APLICAR CODIFICACIÓ A LES COLUMNES (SOBREESCRIVINT)
# ====================================================================

print("\nAplicant codificació Text -> Numèric (sobreescrivint columnes originals)...")

# Columna 'Gender' s'actualitza directament amb els valors codificats (1, 2, 3)
df_treballadors['Gender'] = df_treballadors['Gender'].apply(codificar_sexo).astype('Int64')

# Aplicació de la resta de funcions
df_treballadors['self_employed'] = df_treballadors['self_employed'].apply(codificar_self_employed).astype('Int64')
df_treballadors['family_history'] = df_treballadors['family_history'].apply(codificar_family_history).astype('Int64')
df_treballadors['treatment'] = df_treballadors['treatment'].apply(codificar_treatment).astype('Int64')
df_treballadors['work_interfere'] = df_treballadors['work_interfere'].apply(codificar_work_interfere).astype('Int64')
df_treballadors['no_employees'] = df_treballadors['no_employees'].apply(codificar_no_employees).astype('Int64')
df_treballadors['remote_work'] = df_treballadors['remote_work'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['tech_company'] = df_treballadors['tech_company'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['benefits'] = df_treballadors['benefits'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['care_options'] = df_treballadors['care_options'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['wellness_program'] = df_treballadors['wellness_program'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['seek_help'] = df_treballadors['seek_help'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['anonymity'] = df_treballadors['anonymity'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['leave'] = df_treballadors['leave'].apply(codificar_leave).astype('Int64')
df_treballadors['mental_health_consequence'] = df_treballadors['mental_health_consequence'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['phys_health_consequence'] = df_treballadors['phys_health_consequence'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['coworkers'] = df_treballadors['coworkers'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['supervisor'] = df_treballadors['supervisor'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['mental_health_interview'] = df_treballadors['mental_health_interview'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['phys_health_interview'] = df_treballadors['phys_health_interview'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['mental_vs_physical'] = df_treballadors['mental_vs_physical'].apply(codificar_si_no_otro).astype('Int64')
df_treballadors['obs_consequence'] = df_treballadors['obs_consequence'].apply(codificar_si_no_otro).astype('Int64')

# 4. EXPORTAR EL DATASET FINAL
df_treballadors.to_csv('treballadors_definitiu.csv', index=False)

print("\n[ÈXIT] Dataset de treballadors netejat i codificat a format numèric.")
print(f"Grandària final del dataset: {len(df_treballadors)} registres.")

