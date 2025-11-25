from scipy.stats import chi2_contingency
import pandas as pd


df_estudiants = pd.read_csv('estudiants_med.csv')


df_cat = df_estudiants[['sex', 'psyt', 'health', 'year']]


chi2_table = pd.DataFrame(columns=['Variable 1', 'Variable 2', 'Chi-Square', 'p-value'])


for col1, col2 in [('sex', 'psyt'), ('year', 'health'), ('sex', 'health')]:
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(df_estudiants[col1], df_estudiants[col2]))
    
    
    chi2_table = pd.concat([chi2_table, pd.DataFrame([{'Variable 1': col1, 'Variable 2': col2, 'Chi-Square': chi2, 'p-value': p}])], ignore_index=True)


print("Taula de Chi-Square per a les variables dels estudiants:")
print(chi2_table)


correlation_stud_h_health = df_estudiants['stud_h'].corr(df_estudiants['health'])
print(f"Correlació entre hores d'estudi i salut: {correlation_stud_h_health}")


correlation_age_health = df_estudiants['age'].corr(df_estudiants['health'])
print(f"Correlació entre edat i salut: {correlation_age_health}")

correlation_age_psyt = df_estudiants['age'].corr(df_estudiants['psyt'])
print(f"Correlació entre edat i necessitat de teràpia psicològica: {correlation_age_psyt}")
