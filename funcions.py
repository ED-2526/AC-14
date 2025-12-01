from sklearn.metrics import silhouette_score

def millor_k_silhouette(X, max_k=10):
    scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores.append((k, score))
    return max(scores, key=lambda x: x[1])

k_optimal_sil, sil_score = millor_k_silhouette(df_scaled.drop(columns=["cluster"], errors="ignore"))
print("K òptim segons Silhouette:", k_optimal_sil)
