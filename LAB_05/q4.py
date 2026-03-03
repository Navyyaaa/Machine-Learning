import pandas as pd
import ast
from sklearn.cluster import KMeans
data = pd.read_csv(r"D:\Sem4\ml\lab\assignments\done\LAB05\projectds.csv")
data["embedding"] = data["embedding"].apply(ast.literal_eval)
X = pd.DataFrame(data["embedding"].to_list())
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X)
print("Cluster Labels (first 10):")
print(kmeans.labels_[:10])

print("\nCluster Centers Shape:")
print(kmeans.cluster_centers_.shape)
