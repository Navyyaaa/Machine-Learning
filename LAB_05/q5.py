import pandas as pd
import ast
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
data = pd.read_csv(r"D:\Sem4\ml\lab\assignments\done\LAB05\projectds.csv")
data["embedding"] = data["embedding"].apply(ast.literal_eval)
X = pd.DataFrame(data["embedding"].to_list())
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X)

labels = kmeans.labels_
sil_score = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)
db_score = davies_bouldin_score(X, labels)
print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_score)
