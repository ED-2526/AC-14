

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd



df_estudiants = pd.read_csv('estudiants_med.csv')


X = df_estudiants[['sex', 'stud_h', 'health']]  
y = df_estudiants['psyt']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(class_weight='balanced', random_state=42) #Si no es fa el balanced no va

# Entrenar el model amb les dades d'entrenament
model.fit(X_train, y_train)

# Fer prediccions amb les dades de test
# Aquesta predicció utilitza automàticament el llindar de 0.5
y_pred = model.predict(X_test)

# Avaluar el model amb el report de classificació (Primer informe de la teva imatge)
print("--- Informe amb llindar per defecte (0.5) ---")
print(classification_report(y_test, y_pred, zero_division=1))







 

# 2. Afegir les prediccions a les dades de prova per a la teva anàlisi posterior
# Creem un DataFrame per visualitzar els resultats
resultats_test = X_test.copy()
resultats_test['psyt_real'] = y_test 
resultats_test['psyt_predita'] = y_pred

# 3. Mostrar un resum de les classes identificades
print("--- Distribució de les Classes Predites ---")
print(resultats_test['psyt_predita'].value_counts())

print("\n--- Primers 10 resultats del conjunt de prova ---")
print(resultats_test.head(10))


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- INICI DEL CODI DEL GRÀFIC 3D ---
print("\n" + "="*60)
print("--- 4. GRÀFIC DE DISPERSIÓ 3D DE LES CLASSES PREDITES ---")
print("="*60)

fig = plt.figure(figsize=(12, 10))
# Creem els eixos 3D
ax = fig.add_subplot(111, projection='3d')

# Mapatge dels colors i etiquetes per a la llegenda
colors = {0: 'skyblue', 1: 'red'}
labels = {0: 'Classe 0 Predita (No Teràpia)', 1: 'Classe 1 Predita (Teràpia)'}

# Itera sobre els dos grups de predicció
for i, group in resultats_test.groupby('psyt_predita'):
    ax.scatter(
        group['stud_h'], # Eix X
        group['health'], # Eix Y
        group['sex'],    # Eix Z
        color=colors[i],
        label=labels[i],
        s=80,
        alpha=0.7,
        marker='o'
    )

# Configuració dels títols i etiquetes
ax.set_xlabel('Hores d\'Estudi (stud_h)', fontsize=12)
ax.set_ylabel('Nivell de Salut (health)', fontsize=12)
ax.set_zlabel('Sexe (sex)', fontsize=12)
ax.set_title('Separació de Classes Predites pel Model Logístic (3D)', fontsize=14)

ax.legend(title='Classe Predita')
plt.show()

# --- FINAL DEL CODI DEL GRÀFIC 3D ---