import pandas as pd
import prince
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ===========================
# 1. Carregar dataset
# ===========================
df = pd.read_csv("treballadors_definitiu.csv")

vars_mca = [
    "Gender",
    "self_employed",
    "tech_company",
    "remote_work",
    "no_employees",
    "treatment",
    "family_history",
    "obs_consequence",
]

df_mca = df[vars_mca].copy().fillna("Unknown").astype(str)

print("Variables candidates per MCA:")
print(df_mca.columns.tolist())

# ===========================
# 2. MCA amb TOTES les variables (per mesurar contribucions)
# ===========================
mca = prince.MCA(
    n_components=10,
    random_state=42
).fit(df_mca)

X_mca = mca.row_coordinates(df_mca)
print("\nShape MCA (totes les variables):", X_mca.shape)
print("\nInèrcia explicada per component:")
print(mca.eigenvalues_summary)

# ===========================
# 2b. Contribució per VARIABLE (sumant sobre les primeres components)
# ===========================
col_contrib = mca.column_contributions_        # categories x components

N_COMP_FOR_VAR = 5
cat_total = col_contrib.iloc[:, :N_COMP_FOR_VAR].sum(axis=1)

cat_df = cat_total.to_frame(name="contribucio_categoria")
cat_df["variable"] = cat_df.index.str.split(mca.one_hot_prefix_sep).str[0]

var_contrib = (
    cat_df.groupby("variable")["contribucio_categoria"]
    .sum()
    .sort_values(ascending=False)
)

print(f"\nRÀNQUING DE VARIABLES per contribució total (primeres {N_COMP_FOR_VAR} components):")
print(var_contrib)

# ===========================
# 2c. Escollir TOP variables
# ===========================
TOP_N = 7
top_vars = var_contrib.head(TOP_N).index.tolist()

print(f"\nTOP {TOP_N} variables segons MCA:")
for v in top_vars:
    print("  -", v)

# ===========================
# 3. MCA NOMÉS amb les TOP variables
# ===========================
df_top = df[top_vars].copy().fillna("Unknown").astype(str)

mca_top = prince.MCA(
    n_components=10,
    random_state=42
).fit(df_top)

X_mca_top = mca_top.row_coordinates(df_top)

print("\nShape MCA (només TOP variables):", X_mca_top.shape)
print("\nInèrcia explicada amb només TOP variables:")
print(mca_top.eigenvalues_summary)

# ===========================
# 3b. Elbow + silhouette per triar K
# ===========================
inertia = []
silhouette = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_mca_top.iloc[:, :5])
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_mca_top.iloc[:, :5], labels))

plt.figure(figsize=(10, 4))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow method (MCA TOP vars + K-Means)")
plt.xlabel("Clusters (k)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(K_range, silhouette, marker="o")
plt.title("Silhouette score (MCA TOP vars)")
plt.xlabel("Clusters (k)")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# ===========================
# 4. K-Means final amb K òptim
# ===========================
k_opt = 3   # canvia-ho si vols segons els gràfics
kmeans = KMeans(n_clusters=k_opt, random_state=42)
labels_final = kmeans.fit_predict(X_mca_top.iloc[:, :5])
df["cluster_mca_top"] = labels_final

print("\nMida de cada cluster (model amb TOP variables):")
print(df["cluster_mca_top"].value_counts())

# ===========================
# 5. Perfils categòrics dels clusters (només TOP variables)
# ===========================
perfil_top = df.groupby("cluster_mca_top")[top_vars].agg(lambda x: x.mode()[0])
print("\nPerfils categòrics dels clusters (només TOP variables):")
print(perfil_top)

# ===========================
# 6. GRÀFIC 2D: visualitzar clústers en l'espai MCA
# ===========================
plt.figure(figsize=(9, 7))

# components 0 i 1 de la MCA amb TOP variables
x = X_mca_top[0]
y = X_mca_top[1]

sns.scatterplot(
    x=x,
    y=y,
    hue=df["cluster_mca_top"],
    palette="Set2",
    s=50,
    alpha=0.7,
    edgecolor="k"
)

# centroides dels clústers en aquest espai
centers = kmeans.cluster_centers_[:, :2]
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    s=150,
    marker="X",
    label="Centroides"
)

plt.title("Clústers en l'espai MCA (components 1 i 2)")
plt.xlabel("MCA component 1")
plt.ylabel("MCA component 2")
plt.legend()
plt.tight_layout()
plt.show()
