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
        return np.nan
    
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
        return 1 #Dona = 1
    elif any(keyword in valor for keyword in male_keywords):
        return 0 #Home = 0
    elif any(keyword in valor for keyword in trans_keywords):
        return 0.5 #Altre = 0.5
    else:
        return np.nan
    

def codificar_self_employed(valor):
    "Codifica Yes = 1, NO = 0"
    if pd.isna(valor): 
        return np.nan
    valor = str(valor).lower().strip()
    if valor in ['yes', 'y', 'true', '1']: 
        return int(1)
    elif valor in ['no', 'n', 'false', '0']: 
        return int(0)
    else: 
        return np.nan

def codificar_family_history(valor):
    if pd.isna(valor): 
        return np.nan
    valor = str(valor).lower().strip()
    if valor in ['yes', 'y', 'true', '1']: 
        return int(1)
    elif valor in ['no', 'n', 'false', '0']: 
        return int(0)
    else: 
        return np.nan

def codificar_treatment(valor):
    if pd.isna(valor): 
        return np.nan
    valor = str(valor).lower().strip()
    if valor in ['yes', 'y', 'true', '1']: 
        return int(1)
    elif valor in ['no', 'n', 'false', '0']: 
        return int(0)
    else: 
        return np.nan

# Funció per codificar Work_interfere (Ordinal 0-1)
def codificar_work_interfere(valor):
    if pd.isna(valor):
        return np.nan  # Mantener como NaN por ahora
    
    valor = str(valor).strip()
    
    # BASADO EN TUS DATOS: Never, Rarely, Sometimes, Often (NO hay Always)
    mapping = {
        'Never': 0.0,      
        'Rarely': 0.3333,   
        'Sometimes': 0.6667, 
        'Often': 1.0        
    }
    
    if valor in mapping:
        return mapping[valor]
    
    # Si no está exacto, buscar por contenido
    valor_lower = valor.lower()
    if 'never' in valor_lower:
        return 0.0
    elif 'rarely' in valor_lower:
        return 0.3333
    elif 'sometimes' in valor_lower:
        return 0.6667
    elif 'often' in valor_lower:
        return 1.0
    else:
        return np.nan  # Cualsevol altra = Na

# Funció per codificar No_employees (Ordinal)
def codificar_no_employees(valor):
    if pd.isna(valor): return np.nan
    mapping = {
        '1-5': 0.0,
        '6-25': 0.1429,
        '26-100': 0.2857,
        '100-500': 0.4286,
        '500-1000': 0.5714,
        '1000-5000': 0.7143,
        '5000-10000': 0.8571,
        '10000+': 1.0,
        'More than 1000': 0.7143
    }
    result = mapping.get(valor, np.nan)
    return result

# Funció genèrica per codificar Sí/No/Altre (utilitzada per la majoria de binàries)
def codificar_si_no_otro(valor):
    if pd.isna(valor): 
        return np.nan
    valor = str(valor).lower().strip()
    if valor in ['yes', 'y', 'true', '1']: 
        return int(1)
    elif valor in ['no', 'n', 'false', '0']: 
        return int(0)
    else: 
        return np.nan

# Funció per codificar Leave (Ordinal)
def codificar_leave(valor):
    if pd.isna(valor): 
        return np.nan
    mapping = {
        'Very easy': 0.0,
        'Somewhat easy': 0.3333,
        'Somewhat difficult': 0.6667,
        'Very difficult': 1.0
    }
    return mapping.get(valor, np.nan)

def normalizar_age(age):
    if pd.isna(age):
        return np.nan
    age = float(age)
    if age < 18:
        return 0.0
    elif age > 65:
        return 1.0
    else:
        return (age - 18) / (65 - 18)



# ====================================================================
# 3. APLICAR CODIFICACIÓN
# ====================================================================

print("\nAplicant codificació Text -> Numèric 0-1 (mantenint NaN)...")

# Lista de todas las funciones y sus columnas
funciones_a_aplicar = [
    ('Gender', codificar_sexo),
    ('Age', normalizar_age),
    ('self_employed', codificar_self_employed),
    ('family_history', codificar_family_history),
    ('treatment', codificar_treatment),
    ('work_interfere', codificar_work_interfere),
    ('no_employees', codificar_no_employees),
    ('remote_work', codificar_si_no_otro),
    ('tech_company', codificar_si_no_otro),
    ('benefits', codificar_si_no_otro),
    ('care_options', codificar_si_no_otro),
    ('wellness_program', codificar_si_no_otro),
    ('seek_help', codificar_si_no_otro),
    ('anonymity', codificar_si_no_otro),
    ('leave', codificar_leave),
    ('mental_health_consequence', codificar_si_no_otro),
    ('phys_health_consequence', codificar_si_no_otro),
    ('coworkers', codificar_si_no_otro),
    ('supervisor', codificar_si_no_otro),
    ('mental_health_interview', codificar_si_no_otro),
    ('phys_health_interview', codificar_si_no_otro),
    ('mental_vs_physical', codificar_si_no_otro),
    ('obs_consequence', codificar_si_no_otro)
]

for col, func in funciones_a_aplicar:
    if col in df_treballadors.columns:
        df_treballadors[col] = df_treballadors[col].apply(func)
        print(f" {col}") 

# ====================================================================
# 4. ANÁLISIS DE NANs DESPUÉS DE CODIFICAR
# ====================================================================

print("\n" + "="*70)
print("ANÁLISIS COMPLETO DE VALORES NULOS (DESPUÉS DE CODIFICAR)")
print("="*70)

# Crear análisis detallado
nan_analysis = []
for col in df_treballadors.columns:
    if col not in ['Country', 'state']:  # Country/state no se han codificado
        nan_count = df_treballadors[col].isnull().sum()
        nan_percentage = (nan_count / len(df_treballadors)) * 100
        non_nan_count = len(df_treballadors) - nan_count
        
        # Calcular estadísticas de valores no nulos
        valores_no_nan = df_treballadors[col].dropna()
        if len(valores_no_nan) > 0:
            media = valores_no_nan.mean()
            min_val = valores_no_nan.min()
            max_val = valores_no_nan.max()
        else:
            media = min_val = max_val = np.nan
        
        nan_analysis.append({
            'variable': col,
            'n_nans': nan_count,
            '%_nans': nan_percentage,
            'n_validos': non_nan_count,
            'media_valores_validos': media,
            'min_valido': min_val,
            'max_valido': max_val,
            'rango': f"[{min_val:.2f}-{max_val:.2f}]" if not pd.isna(min_val) else "N/A"
        })

# Convertir a DataFrame y ordenar
df_nan_analysis = pd.DataFrame(nan_analysis).sort_values('%_nans', ascending=False)

print("\n RESUMEN POR VARIABLE (ordenado por % NANs):")
print(df_nan_analysis[['variable', 'n_nans', '%_nans', 'n_validos', 'media_valores_validos', 'rango']].to_string())

# Totales
print(f"\n ESTADÍSTICAS GLOBALES:")
total_celdas = len(df_treballadors) * len(df_treballadors.columns)
total_nans = df_treballadors.isnull().sum().sum()
print(f"  Total de celdas en dataset: {total_celdas:,}")
print(f"  Total de NANs: {total_nans:,} ({total_nans/total_celdas*100:.1f}% de todas las celdas)")

# Variables críticas (con muchos NANs)
print(f"\n VARIABLES CON MÁS DEL 10% DE NANs:")
variables_criticas = df_nan_analysis[df_nan_analysis['%_nans'] > 10]
for _, row in variables_criticas.iterrows():
    print(f"  • {row['variable']:25} - {row['n_nans']:4d} NANs ({row['%_nans']:.1f}%)")

# Variables limpias
print(f"\n VARIABLES CON MENOS DEL 5% DE NANs:")
variables_limpias = df_nan_analysis[df_nan_analysis['%_nans'] < 5]
for _, row in variables_limpias.iterrows():
    print(f"  • {row['variable']:25} - {row['n_nans']:4d} NANs ({row['%_nans']:.1f}%)")

# ====================================================================
# 5. GUARDAR DATASET CON NANs
# ====================================================================

df_treballadors.to_csv('treballadors_definitiu.csv', index=False)

print("\n" + "="*70)
print("DATASET GUARDADO CON NANs PARA ANÁLISIS")
print("="*70)
print(f"Archivo: 'treballadors_definitiu_CON_NANs.csv'")
print(f"Registros: {len(df_treballadors)}")
print(f"Variables: {len(df_treballadors.columns)}")
print(f"NANs totales: {total_nans:,} ({total_nans/total_celdas*100:.1f}% de celdas)")
print("\n AHORA PUEDES DECIDIR POR VARIABLE:")
print("  1. Variables con >20% NANs, Considerar ELIMINAR")
print("  2. Variables clave con 5-20% NANs, IMPUTAR con media/moda")
print("  3. Variables con <5% NANs, ELIMINAR FILAS o IMPUTAR")
print("="*70)