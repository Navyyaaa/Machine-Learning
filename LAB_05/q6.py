import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
data = pd.read_csv(r"D:\Sem4\ml\lab\assignments\done\LAB05\projectds.csv")

data["embedding"] = data["embedding"].apply(ast.literal_eval)
X = pd.DataFrame(data["embedding"].to_list())
k_values = range(2, 11)

sil_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    labels = kmeans.labels_

    sil_scores.append(silhouette_score(X, labels))
    ch_scores.append(calinski_harabasz_score(X, labels))
    db_scores.append(davies_bouldin_score(X, labels))
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(k_values, sil_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")

plt.subplot(1,3,2)
plt.plot(k_values, ch_scores, marker='o')
plt.title("CH Score vs k")
plt.xlabel("k")
plt.ylabel("CH Score")

plt.subplot(1,3,3)
plt.plot(k_values, db_scores, marker='o')
plt.title("DB Index vs k")
plt.xlabel("k")
plt.ylabel("DB Index")

plt.tight_layout()
plt.show()
